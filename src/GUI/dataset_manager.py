
#!/usr/bin/env python3
# encoding: utf-8
"""
    Handles the loading of dataset data to be used later by the GUY
"""


import os
import sys
import time
import shutil
import inspect
import re


from datetime import datetime

import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import concurrent.futures
import traceback

import csv
from tqdm import tqdm

if __name__ == "__main__":
    import sys
    sys.path.append('./src')

# Import paths from centralized config (single source of truth)
from config import yolo_output_path, yolo_output_path_2

from GUI.compute_plot_data import compute_plot_data
from utils import parseYaml, dumpYaml
from utils import log, bcolors

data_file_name = "results.yaml"
ignore_file_name = "EEHA_GUI_IGNORE" # If a file is found in path with this name the folder would be ignored
cache_path = f"{os.getenv('HOME')}/.cache/eeha_gui_cache"
cache_extension = '.yaml.cache'
test_key_clean = [r'_4090[0-9]{0,2}', r'_3090[0-9]{0,2}', '_GPU3090', '_A30', r'_[0-9]{8,9}', r'_[0-9]{6}', r'_GPU[0-1]'] # Path tags to be cleared from key (merges tests from different GPUs). Leave empty for no merging


_KNOWN_CONDITIONS = ('day', 'night', 'all')
_KNOWN_DATASETS = ('kaist', 'llvip')


def parse_dataset_name(title):
    """
    Parse a dataset title like 'kaist_wavelet_max_day' into (dataset, method, condition).

    Heuristic: first token is the dataset (kaist, llvip…); if the last token is a known
    condition (day/night/all) it's split off; everything else joined by '_' is the method.
    Falls back gracefully when the format doesn't match.

    Examples:
        'kaist_wavelet_max_day'   → ('kaist', 'wavelet_max', 'day')
        'kaist_sobel_weighted_day'→ ('kaist', 'sobel_weighted', 'day')
        'kaist_pca_day'           → ('kaist', 'pca', 'day')
        'kaist_split_late_4ch_day'→ ('kaist', 'split_late_4ch', 'day')
        'kaist_visible'           → ('kaist', 'visible', '')
    """
    parts = title.split('_')
    if not parts:
        return ('', '', '')
    dataset = parts[0]
    if len(parts) >= 2 and parts[-1] in _KNOWN_CONDITIONS:
        condition = parts[-1]
        method = '_'.join(parts[1:-1])
    else:
        condition = ''
        method = '_'.join(parts[1:])
    return (dataset, method, condition)


def extract_common_dataset_condition(info_dict, keys):
    """
    Return ((dataset, condition), is_mixed) describing the dataset/condition tags
    shared by every key in `keys`.

    - dataset/condition: the single shared value, or None if no key produced one.
    - is_mixed: True when at least one of the two saw conflicting values across
      keys (i.e. the selection is heterogeneous). Callers that want to omit a
      partial suptitle on mixed selections can check this flag.

    Accepts both leaf trial keys (present in info_dict) and group prefixes
    (e.g. 'no_equalization/variance_kaist_curvelet_day') used by variance tabs.
    """
    datasets, conditions = set(), set()
    for k in keys:
        candidates = []
        info = info_dict.get(k)
        if info is None:
            for ik, iinfo in info_dict.items():
                if ik.startswith(f"{k}/"):
                    info = iinfo
                    break
        if info:
            candidates.append(info.get('title', ''))
            candidates += [seg.replace('variance_', '') for seg in info.get('group_path', [])]
        # Always try the raw key string too — covers synthetic/percentile keys
        # and bare group prefixes when info_dict has no match.
        candidates += [seg.replace('variance_', '') for seg in str(k).split('/')]

        ds_found, cond_found = None, None
        for c in candidates:
            ds, _, cond = parse_dataset_name(c)
            if ds in _KNOWN_DATASETS and ds_found is None:
                ds_found = ds
            if cond in _KNOWN_CONDITIONS and cond_found is None:
                cond_found = cond
            if ds_found and cond_found:
                break

        if ds_found:
            datasets.add(ds_found)
        if cond_found:
            conditions.add(cond_found)

    is_mixed = len(datasets) > 1 or len(conditions) > 1
    ds = next(iter(datasets)) if len(datasets) == 1 else None
    cond = next(iter(conditions)) if len(conditions) == 1 else None
    return (ds, cond), is_mixed


def parseCSV(file_path):
    with open(file_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        headers = [header.strip() for header in next(csv_reader)]
        data_by_columns = {header: [] for header in headers}

        for row in csv_reader:
            for header, value in zip(headers, row):
                data_by_columns[header].append(value.strip())

        return data_by_columns
    
    
# Gets filtered data from YAML file
def getResultsYamlData(dataset):
    data = parseYaml(dataset['path'])
    data_filtered = {}
    if not data:
        log(f"[{inspect.currentframe().f_code.co_name}] Empty data in {dataset['key']}).", bcolors.ERROR)
        return {}

    data_filtered = compute_plot_data(data, dataset)
    return data_filtered
                        

def getCSVData(dataset):
    # sibling files derived from the dir (robust to results.yaml or results_reconstructed.yaml)
    csv_path = os.path.join(os.path.dirname(dataset['path']), 'results.csv')
    if os.path.exists(csv_path):
        data= parseCSV(csv_path)
        return {'csv_data': data}
    else:
        log(f"\t· Could not parse CSV file associated to {dataset['key']}", bcolors.ERROR)
        return {}
    
def getArgsYamlData(dataset):
    arg_path = os.path.join(os.path.dirname(dataset['path']), 'args.yaml')
    data_filtered = {}
    if os.path.exists(arg_path):
        try:
            data = parseYaml(arg_path)
            data_filtered = {'batch': data['batch'], 'deterministic': data['deterministic']}
        except KeyError as e:
            log(f"[{inspect.currentframe().f_code.co_name}] Missing key in Args data dict ({dataset['key']}): {e}", bcolors.ERROR)
    else:
        log(f"\t· Could not parse Args file file associated to {dataset['key']}", bcolors.ERROR)
    
    return data_filtered

# Wrap function to be paralelized
def background_load_data(dataset_key_tuple):
    key, dataset, update_cache, load_from_cache = dataset_key_tuple
    
    # When loading from cache, use the actual path found by find_cache_file
    # instead of reconstructing from key (avoids key↔path mismatches)
    if load_from_cache and 'path' in dataset and dataset['path'].endswith(cache_extension):
        filename = dataset['path']
    else:
        filename = f"{cache_path}/{key}{cache_extension}".replace('//', '/')
    
    cache_exists = os.path.exists(filename)
    data = None
    
    # Use cache if: (exists and not forcing update) OR (load_from_cache mode and exists)
    if (cache_exists and not update_cache) or (load_from_cache and cache_exists):
        data = parseYaml(filename)
        # Handle corrupted or empty cache files
        if data is None or data == {}:
            if load_from_cache:
                log(f"Cache file corrupted or empty for {key}, skipping", bcolors.WARNING)
                return None
            else:
                # Try to regenerate from raw data
                log(f"Cache file corrupted for {key}, regenerating from raw data", bcolors.WARNING)
                data = None  # Force regeneration below
        # log(f"Loaded data from cache file in {filename}")
    
    if data is None:
        if load_from_cache and not cache_exists:
            # Cache requested but doesn't exist - skip this dataset
            log(f"Cache file not found for {key}, skipping (use --force_update_cache to regenerate)", bcolors.WARNING)
            return None
        elif load_from_cache:
            # Already logged above as corrupted
            return None
        else:
            data = getResultsYamlData(dataset)
            if data is None or data == {}:
                log(f"Detected error in data loader for {key} dataset")
                return None
            data.update(getCSVData(dataset))
            data.update(getArgsYamlData(dataset))
            log(f"Reloaded data from RAW data for {key} dataset")

    # Update cache data from data currently parsed (skip when only reading cache)
    if not load_from_cache and data is not None and data != {}:
        cache_key_path = os.path.dirname(filename)
        os.makedirs(cache_key_path, exist_ok=True)
        dumpYaml(filename, data)
    
    return data

"""
    :param: ignored ignore or not the folders with ignore files. False to process all
        even with ignore file
"""
def find_results_file(search_path_list = [yolo_output_path], file_name = data_file_name, ignored = True):
    global test_key_clean

    dataset_info = {}
    for search_path in search_path_list:
        log(f"Search all {file_name} files in {search_path}")
        for root, dirs, files in os.walk(search_path):

            if ignore_file_name in files and ignored:
                log(f"[find_results_file] Path with {ignore_file_name} file is set to be ignored: {root}.", bcolors.WARNING)
                dirs[:] = [] # Clear subdir list to avoid getting into them
                continue

            if file_name in files:
                # Prefer reconstructed results when present. results.yaml lost its
                # pr_data_0 block to the dumpYaml bug (~2026-05-16); results_reconstructed.yaml
                # is a complete superset (original metadata + recomputed pr_data_0).
                # See scripts/reconstruct_results.py.
                recon_name = "results_reconstructed.yaml"
                abs_path = os.path.join(root, recon_name if recon_name in files else file_name)
                if "validate" in abs_path: # Avoid validation tests, only training
                    log(f"[find_results_file] Validate tests are to be ignored: {abs_path}.", bcolors.WARNING)
                    continue
                group_path = root.replace(search_path,'').strip('/').split("/")[:-2]
                name = abs_path.split("/")[-2]
                title = abs_path.split("/")[-2]
                model = abs_path.split("/")[-3]
                
                for clear_pattern in test_key_clean:
                    # model = model.replace(clear_tag, "")
                    model = re.sub(clear_pattern, "", model)
                    title = re.sub(clear_pattern, "", title)
                key = f"{'/'.join(group_path)}/{model}/{name}".replace('//','/')
                dataset, method, _ = parse_dataset_name(title)
                # ax_label = f"{dataset.title()} {method.upper()}".strip() + f" ({model.replace('_sameseed','')})"
                ## Plot by dataset, so its not needed in label
                ax_label = f"{method.upper()}".strip() + f" ({model.replace('_sameseed','')})"
                dataset_info[key] = {'name': name, 'path': abs_path, 'model': model, 'key': key, 'title': f"{title}", 'label': f'{ax_label}', 'group_path': group_path}

    ## Order dataset by name
    myKeys = list(dataset_info.keys())
    myKeys.sort()
    dataset_info = {i: dataset_info[i] for i in myKeys}
    return dataset_info

"""
    Equivalent to find_results_file when data is to be loaded directly from cache
"""
def find_cache_file(search_path = cache_path, file_name = cache_extension):
    global test_key_clean
    log(f"Search all {file_name} files in {search_path}")

    dataset_info = {}
    for root, dirs, files in os.walk(search_path):
        for file in files:
            if file_name in file:
                abs_path = os.path.join(root, file)
                key_name = abs_path.replace(file_name, "").replace(search_path, "").strip('/')
                name = key_name.split("/")[-1]
                title = key_name.split("/")[-1]
                model = key_name.split("/")[-2]
                group_path = [] if len(key_name.split("/")) < 3 else key_name.split("/")[:-2]
                for clear_pattern in test_key_clean:
                    # model = model.replace(clear_tag, "")
                    model = re.sub(clear_pattern, "", model)
                    title = re.sub(clear_pattern, "", title)
                key = '/'.join([*group_path, model, name])
                dataset, method, _ = parse_dataset_name(title)
                ax_label = f"{method.upper()}".strip() + f" ({model.replace('_sameseed','')})"
                dataset_info[key] = {'name': name, 'path': abs_path, 'model': model, 'key': key, 'title': f"{title}", 'label': f'{ax_label}', 'group_path': group_path}

    myKeys = list(dataset_info.keys())
    myKeys.sort()
    dataset_info = {i: dataset_info[i] for i in myKeys}
    return dataset_info


from enum import Enum
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Any

class LoadState(Enum):
    NOT_LOADED = 0
    QUEUED = 1  
    LOADING = 2
    LOADED = 3
    FAILED = 4

@dataclass(order=True)
class LoadJob:
    priority: int
    key: str = field(compare=False)
    info: dict = field(compare=False)


class DataSetHandler:
    """
    Lazy-loading dataset handler with background cache generation.
    
    - Discovery is instant (just finds paths)
    - Data is loaded on-demand when __getitem__ is called
    - Background worker generates caches with low priority
    - Priority queue ensures requested datasets load first
    """
    
    PRIORITY_HIGH = 0    # User requested this dataset
    PRIORITY_NORMAL = 10 # Background generation
    
    def __init__(self, update_cache=True, search_path_list=[yolo_output_path, yolo_output_path_2],
                 load_from_cache=False, lazy_load=True, start_background=True):
        self.update_cache = update_cache
        self.load_from_cache = load_from_cache
        self.lazy_load = lazy_load
        self.start_background = start_background
        
        # Discovery phase (fast)
        t_discovery = time.time()
        if load_from_cache:
            self.dataset_info = find_cache_file()
        else:
            global cache_path
            if update_cache and os.path.exists(cache_path):
                shutil.rmtree(cache_path)
                log(f"[{self.__class__.__name__}] Cleared previous cache files to be recached.")
            os.makedirs(cache_path, exist_ok=True)
            self.dataset_info = find_results_file(search_path_list)
        log(f"[{self.__class__.__name__}]Discovery completed: {len(self.dataset_info)} datasets found in {time.time() - t_discovery:.2f}s.")
        
        # State tracking
        self._cache = {}  # key -> loaded data
        self._state = {key: LoadState.NOT_LOADED for key in self.dataset_info}
        self._futures = {}  # key -> Future
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        
        # Work queue with priority
        self._job_queue = PriorityQueue()
        
        # For lazy loading use ThreadPoolExecutor (avoids fork+Qt deadlock)
        # For blocking mode use ProcessPoolExecutor (better for CPU-intensive parsing)
        max_workers = min(4, os.cpu_count() or 2)
        self._executor = None  # Created lazily based on mode
        self._max_workers = max_workers
        
        # Track incomplete datasets
        self.incomplete_dataset = {}
        
        # Stats
        self._loaded_count = 0
        self._failed_count = 0
        
        if lazy_load:
            if start_background:
                # Start background generation (low priority)
                self._start_background_generation()
                log(f"Lazy loading enabled: {len(self.dataset_info)} datasets discovered, loading on demand.")
            else:
                log(f"Lazy loading enabled (no background): {len(self.dataset_info)} datasets discovered, on-demand only.")
        else:
            # Legacy mode: load everything at startup
            self._load_all_blocking()
    
    def _start_background_generation(self):
        """Queue all datasets for background loading with low priority."""
        for key, info in self.dataset_info.items():
            self._job_queue.put(LoadJob(self.PRIORITY_NORMAL, key, info))
            with self._lock:
                if self._state[key] == LoadState.NOT_LOADED:
                    self._state[key] = LoadState.QUEUED
        
        # Start worker threads
        n_workers = 2
        log(f"[{self.__class__.__name__}] Starting {n_workers} background worker threads for {self._job_queue.qsize()} queued datasets.")
        for i in range(n_workers):
            thread = threading.Thread(target=self._background_worker, name=f"bg-loader-{i}", daemon=True)
            thread.start()
    
    def _background_worker(self):
        """Process jobs from the priority queue."""
        worker_name = threading.current_thread().name
        worker_count = 0
        t_worker_start = time.time()
        
        while True:
            try:
                job = self._job_queue.get(timeout=1.0)
            except:
                # Check if there's still work to do
                with self._lock:
                    pending = [k for k, s in self._state.items() 
                              if s in (LoadState.NOT_LOADED, LoadState.QUEUED)]
                if not pending:
                    break
                continue
            
            key = job.key
            info = job.info
            
            with self._lock:
                # Skip if already loaded or being loaded
                if self._state[key] in (LoadState.LOADED, LoadState.LOADING, LoadState.FAILED):
                    self._job_queue.task_done()
                    continue
                self._state[key] = LoadState.LOADING
            
            t_start = time.time()
            try:
                # Load directly in worker thread (avoids fork+Qt deadlock)
                result = background_load_data((key, info, self.update_cache, self.load_from_cache))
                
                with self._lock:
                    if result is not None:
                        self._cache[key] = result
                        self._state[key] = LoadState.LOADED
                        self._loaded_count += 1
                        worker_count += 1
                    else:
                        self._state[key] = LoadState.FAILED
                        self._failed_count += 1
                        # Remove from dataset_info
                        if key in self.dataset_info:
                            del self.dataset_info[key]
                    
                    total_done = self._loaded_count + self._failed_count
                    total = total_done + sum(1 for s in self._state.values() 
                                             if s in (LoadState.NOT_LOADED, LoadState.QUEUED, LoadState.LOADING))
                    
                    # Notify any waiting threads
                    self._condition.notify_all()
                
                # Log progress every 50 datasets or on slow loads (>2s)
                elapsed = time.time() - t_start
                if worker_count % 50 == 0 or elapsed > 2.0:
                    log(f"[{self.__class__.__name__}] [{worker_name}] Progress: {total_done}/{total} datasets loaded ({self._failed_count} failed) - last: {key} ({elapsed:.1f}s)")
                    
            except Exception as e:
                log(f"[{self.__class__.__name__}] [{worker_name}] Error loading {key}: {e}", bcolors.ERROR)
                with self._lock:
                    self._state[key] = LoadState.FAILED
                    self._failed_count += 1
                    self._condition.notify_all()
            
            finally:
                self._job_queue.task_done()
        
        elapsed_total = time.time() - t_worker_start
        log(f"[{self.__class__.__name__}] [{worker_name}] Finished: processed {worker_count} datasets in {elapsed_total:.1f}s.")
    
    def _load_single(self, key: str) -> Any:
        """Load a single dataset with high priority. Blocks until loaded."""
        with self._lock:
            state = self._state.get(key)
            
            # Already loaded
            if state == LoadState.LOADED:
                return self._cache[key]
            
            # Failed previously
            if state == LoadState.FAILED:
                return None
            
            # Already loading - wait for it
            if state == LoadState.LOADING:
                log(f"[{self.__class__.__name__}] [on-demand] Waiting for {key} (already being loaded by background worker)...")
                t_wait = time.time()
                while self._state[key] == LoadState.LOADING:
                    self._condition.wait()
                log(f"[{self.__class__.__name__}] [on-demand] {key} ready after {time.time() - t_wait:.2f}s wait.")
                return self._cache.get(key)
            
            # Not yet loading - submit with high priority
            info = self.dataset_info.get(key)
            if not info:
                log(f"[{self.__class__.__name__}] [on-demand] Key not found in dataset_info: {key}", bcolors.WARNING)
                return None
            
            self._state[key] = LoadState.LOADING
        
        # Load synchronously (high priority = block and wait)
        log(f"[{self.__class__.__name__}] [on-demand] Loading {key} synchronously (high priority)...")
        t_start = time.time()
        try:
            result = background_load_data((key, info, self.update_cache, self.load_from_cache))
            elapsed = time.time() - t_start
            
            with self._lock:
                if result is not None:
                    self._cache[key] = result
                    self._state[key] = LoadState.LOADED
                    self._loaded_count += 1
                    log(f"[{self.__class__.__name__}] [on-demand] Loaded {key} in {elapsed:.2f}s.")
                else:
                    self._state[key] = LoadState.FAILED
                    self._failed_count += 1
                    log(f"[{self.__class__.__name__}] [on-demand] Failed to load {key} after {elapsed:.2f}s.", bcolors.WARNING)
                self._condition.notify_all()
            
            return result
            
        except Exception as e:
            log(f"[{self.__class__.__name__}] [on-demand] Error loading {key}: {e}", bcolors.ERROR)
            with self._lock:
                self._state[key] = LoadState.FAILED
                self._failed_count += 1
                self._condition.notify_all()
            return None
    
    def _load_all_blocking(self):
        """Legacy mode: load all datasets at startup with progress bar."""
        total = len(self.dataset_info)
        log(f"Loading {total} datasets...")
        
        # Use ProcessPoolExecutor for blocking mode (better parallelism for CPU-bound parsing)
        self._executor = ProcessPoolExecutor(max_workers=self._max_workers)
        
        with tqdm(total=total, desc="Loading datasets", unit="dataset") as pbar:
            futures = {}
            for key, info in self.dataset_info.items():
                futures[key] = self._executor.submit(
                    background_load_data,
                    (key, info, self.update_cache, self.load_from_cache)
                )
            
            for key, future in futures.items():
                try:
                    result = future.result()
                    if result is not None:
                        self._cache[key] = result
                        self._state[key] = LoadState.LOADED
                        self._loaded_count += 1
                    else:
                        self._state[key] = LoadState.FAILED
                        self._failed_count += 1
                        if key in self.dataset_info:
                            del self.dataset_info[key]
                except Exception as e:
                    log(f"Error loading {key}: {e}", bcolors.ERROR)
                    self._state[key] = LoadState.FAILED
                    self._failed_count += 1
                finally:
                    pbar.update(1)
        
        log(f"[{self.__class__.__name__}] Blocking load completed: {self._loaded_count} datasets loaded, {self._failed_count} failed.")
        os.system('notify-send "GUI - Dataset manager" "Data loading completed"')
    
    def get_load_status(self) -> dict:
        """Return loading statistics."""
        with self._lock:
            counts = {}
            for state in LoadState:
                counts[state.name] = sum(1 for s in self._state.values() if s == state)
            return counts
    
    def is_fully_loaded(self) -> bool:
        """Check if all datasets are loaded."""
        with self._lock:
            return all(s in (LoadState.LOADED, LoadState.FAILED) 
                      for s in self._state.values())
    
    def wait_for_all(self, timeout=None):
        """Wait for all datasets to finish loading."""
        start = time.time()
        with self._lock:
            while not self.is_fully_loaded():
                remaining = None
                if timeout:
                    elapsed = time.time() - start
                    if elapsed >= timeout:
                        return False
                    remaining = timeout - elapsed
                self._condition.wait(remaining)
        return True
    
    # ===== Public API (compatible with old interface) =====
    
    def getInfo(self):
        return self.dataset_info

    def keys(self):
        return self.dataset_info.keys()
    
    def markAsIncomplete(self, key):
        """Mark dataset as incomplete (has issues)."""
        if key in self._cache:
            self.incomplete_dataset[key] = self.dataset_info.pop(key, None)
            del self._cache[key]
            log(f"[{self.__class__.__name__}] Marked {key} as incomplete.", bcolors.WARNING)

    def reloadIncomplete(self):
        """Reload datasets marked as incomplete."""
        for key, info in self.incomplete_dataset.items():
            self.dataset_info[key] = info
            self._state[key] = LoadState.NOT_LOADED
            self._job_queue.put(LoadJob(self.PRIORITY_HIGH, key, info))
        self.incomplete_dataset.clear()
    
    def __getitem__(self, key):
        """Get dataset data. Loads on-demand if not cached."""
        if key in self.incomplete_dataset:
            return None
        
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        
        # Load on demand
        return self._load_single(key)
    
    def inject(self, key, data, info):
        """Inject synthetic computed data (e.g. percentile estimates) into the cache."""
        with self._lock:
            self._cache[key] = data
        self.dataset_info[key] = info

    def eject(self, key):
        """Remove a previously injected synthetic key."""
        with self._lock:
            self._cache.pop(key, None)
        self.dataset_info.pop(key, None)

    def __contains__(self, key):
        return key in self.dataset_info
    
    def __len__(self):
        return len(self.dataset_info)
    
    def __delitem__(self, key):
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            if key in self.dataset_info:
                info = self.dataset_info.pop(key)
                return info
        raise KeyError(f'Key {key} not found.')
    
    def __del__(self):
        if hasattr(self, '_executor') and self._executor:
            self._executor.shutdown(wait=False)

    def get_training_data_summary(self):
        summary = "Training Data Summary:\n\n"
        for key in self.dataset_info.keys():
            data = self.__getitem__(key)
            print(f"[get_training_data_summary] Processing {key = }: {data = }")
            if data is None:
                summary += f"· {key}: Incomplete dataset, data could not be loaded.\n"
                continue

            summary += f"· {key}:\n"
            try:
                batch = data.get('batch', 'N/A')
                deterministic = data.get('deterministic', 'N/A')
                summary += f"    - Batch size: {batch}\n"
                summary += f"    - Deterministic: {deterministic}\n"
                
                if 'results' in data:
                    results = data['results']
                    for metric, values in results.items():
                        if isinstance(values, list) and values:
                            final_value = values[-1]
                            summary += f"    - Final {metric}: {final_value}\n"
                        else:
                            summary += f"    - {metric}: No data available\n"
                else:
                    summary += "    - No results data available\n"
            except Exception as e:
                summary += f"    - Error retrieving data: {e}\n"
        return summary


"""
    Main execution allows to cache data ind advance without loading the whole gui :)
"""
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run dataset manager to gather all data from yaml.")
    parser.add_argument('--update_cache', action='store_true', help='If true all cache data is updated from scratch.')

    # Parsear los argumentos de la línea de comandos
    args = parser.parse_args()

    update_cache = args.update_cache
    dataset_handler = DataSetHandler(update_cache)