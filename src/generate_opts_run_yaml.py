#!/usr/bin/env python3
# encoding: utf-8

"""
For each directory tries to get an args.yaml and a results.yaml file. If both exist gathers some data from them to be outputed in a new
run_opts.yaml file in the same path.

Output format should be like this:
    batch: 16
    cache: disk
    clist: [night]
    dataset: null
    deterministic: true
    device: null
    dformat: kaist_80_20
    distortion_correct: true
    iterations: 5
    mlist: [yoloCh3m.yaml]
    olist: [superpixel]
    path_name: rgb_equalization/variance_kaist_superpixel_night
    pretrained: false
    relabeling: true
    resume: false
    rgb_eq: clahe
    run_mode: [train]
    test_name: null
    thermal_eq: none
"""


import os
import csv
import re
import shutil
import yaml
from pathlib import Path
import tabulate
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threadedprocess import ThreadedProcessPoolExecutor
from tqdm import tqdm
from utils.yaml_utils import parseYaml, dumpYaml
from utils.log_utils import log, bcolors

FORCE_REGENERATE = False
BASE_SEARCH_PATH = "/home/arvc/eeha/kaist-cvpr15/runs/detect"
REGENERATE_ERROR_RUN_OPTS = False
ANALYZE_DIRECTORY_STRUCTURE = True
FIX_DIRECTORY_STRUCTURE = False
FIX_EQUALIZATION_MISMATCHES = False
ANOMALY_REPORT_CSV = "run_opts_anomaly_report.csv"
ANOMALY_REPORT_TXT = "run_opts_anomaly_report.txt"
DETAILED_COUNTS_CSV = "run_opts_detailed_counts.csv"
EQ_ROOTS = {'no_equalization', 'th_equalization', 'rgb_equalization', 'rgb_th_equalization'}
IGNORED_TOP_LEVEL_DIRS = {"other"}
IGNORED_VARIANCE_GROUP_DIRS = {"variance_llvip_fa_patches"}


def _is_ignored_relative_path(root_path, path_obj):
    """Ignore paths that belong to excluded top-level folders under BASE_SEARCH_PATH."""
    try:
        rel_parts = Path(path_obj).relative_to(Path(root_path)).parts
    except Exception:
        return False

    if not rel_parts:
        return False

    if rel_parts[0] in IGNORED_TOP_LEVEL_DIRS:
        return True

    # Also ignore selected variance groups under equalization roots:
    # <eq_root>/<variance_group>/...
    if (
        len(rel_parts) >= 2
        and rel_parts[0] in EQ_ROOTS
        and rel_parts[1] in IGNORED_VARIANCE_GROUP_DIRS
    ):
        return True

    return False


def _infer_condition_from_text(text, fallback=None):
    value = (text or "").lower()
    if "day" in value:
        return "day"
    if "night" in value:
        return "night"
    return fallback


def _infer_dataset_split_from_texts(*texts):
    joined = " ".join([str(t or "") for t in texts]).lower()
    dataset_name = 'unknown'
    if 'kaist' in joined:
        dataset_name = 'kaist'
    elif 'llvip' in joined:
        dataset_name = 'llvip'
    elif 'coco' in joined:
        dataset_name = 'coco'

    split_name = 'unknown_split'
    for candidate in ('90_10', '80_20', '70_30'):
        if candidate in joined:
            split_name = candidate
            break

    return dataset_name, split_name


def _normalize_dataset_tag(dataset, *fallback_texts):
    dataset = str(dataset or '')
    if dataset and not dataset.startswith('unknown_'):
        return dataset

    dataset_name, split_name = _infer_dataset_split_from_texts(*fallback_texts)
    return f"{dataset_name}_{split_name}"


def _normalize_condition(condition, *fallback_texts):
    value = str(condition or '')
    lowered = value.lower()
    if lowered in ('day', 'night'):
        return lowered

    if lowered.startswith('unknown_condition'):
        inferred = _infer_condition_from_text(" ".join([str(t or '') for t in fallback_texts]))
        if inferred:
            return inferred

    return value


def _infer_dataset_tag(args_data, results_data):
    dataset_tag = results_data.get('dataset_tag')
    if dataset_tag and not str(dataset_tag).startswith('unknown_'):
        return dataset_tag

    dataset_info = results_data.get('dataset_info')
    train_path = ''
    if isinstance(dataset_info, dict):
        train_data = dataset_info.get('train')
        if isinstance(train_data, list) and train_data:
            train_path = train_data[0]
        elif isinstance(train_data, str):
            train_path = train_data

    dataset_name, split_name = _infer_dataset_split_from_texts(
        args_data.get('data', ''),
        args_data.get('name', ''),
        train_path,
        dataset_tag,
    )
    return f"{dataset_name}_{split_name}"


def _infer_option_executed(args_data, results_data):
    dataset_info = results_data.get('dataset_info')
    if isinstance(dataset_info, dict):
        train_path = dataset_info.get('train')
        if isinstance(train_path, list) and train_path and isinstance(train_path[0], str):
            return train_path[0].split('/')[-1]

    # Fallback for old/partial results.yaml files.
    name = Path(str(args_data.get('name', '') or '')).name
    tokens = [t for t in name.split('_') if t]
    if len(tokens) >= 2:
        return tokens[1]
    return 'unknown_option'


def _normalize_fusion_name(fusion, path_name, condition):
    """Keep the base fusion stable and only extend with meaningful suffixes."""
    if not fusion:
        return 'unknown_fusion'

    fusion = str(fusion)
    base = Path(path_name or '').name
    tokens = [t for t in base.split('_') if t]
    if fusion not in tokens:
        return fusion

    idx = tokens.index(fusion)
    tail = tokens[idx + 1:]
    if not tail:
        return fusion

    noise_tokens = {
        'llvip', 'kaist', 'coco',
        'rgb', 'th', 'no', 'equalization',
        'rgbt', 'rgb_th'
    }
    cleaned_tail = [
        t for t in tail
        if t not in noise_tokens and t not in (condition, 'day', 'night')
    ]
    if not cleaned_tail:
        return fusion

    allowed_suffix_starts = {
        'patches', 'max', 'weighted', 'v2', 'v3',
        'split', 'late', 'early', 'middle',
        '4ch', '3ch', '2ch', '1ch'
    }
    if cleaned_tail[0] in allowed_suffix_starts:
        return '_'.join([fusion] + cleaned_tail)

    return fusion


def _normalize_test_path(test_path):
    """Collapse accidental nested equalization path segments produced by old runs."""
    parts = Path(test_path).parts
    eq_roots = {'no_equalization', 'th_equalization', 'rgb_equalization', 'rgb_th_equalization'}
    if len(parts) >= 3 and parts[0] in eq_roots and parts[1].startswith('variance_'):
        nested = parts[2]
        if nested.startswith('variance_') and nested.endswith('equalization'):
            return '/'.join([parts[0], parts[1]])
    return test_path


def _expected_equalization_type(rgb_eq, thermal_eq):
    rgb = str(rgb_eq or 'none')
    thermal = str(thermal_eq or 'none')
    if rgb == 'none' and thermal == 'none':
        return 'no_equalization'
    if rgb != 'none' and thermal == 'none':
        return 'rgb_equalization'
    if rgb == 'none' and thermal != 'none':
        return 'th_equalization'
    return 'rgb_th_equalization'


def _equalization_type_from_text(text):
    value = str(text or '').lower()
    if 'rgb_th_equalization' in value:
        return 'rgb_th_equalization'
    if 'th_equalization' in value:
        return 'th_equalization'
    if 'rgb_equalization' in value:
        return 'rgb_equalization'
    if 'no_equalization' in value:
        return 'no_equalization'
    return None


def _extract_results_hints_fast(results_path):
    """Fast text extraction of only the keys needed for equalization mismatch checks."""
    rgb_eq = None
    thermal_eq = None
    test_text = None
    name_text = None

    try:
        with open(results_path, 'r', encoding='utf-8', errors='ignore') as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line or line.startswith('#'):
                    continue

                if rgb_eq is None and line.startswith('rgb_equalization:'):
                    rgb_eq = line.split(':', 1)[1].strip()
                    continue

                if thermal_eq is None and line.startswith('thermal_equalization:'):
                    thermal_eq = line.split(':', 1)[1].strip()
                    continue

                if test_text is None and line.startswith('test:'):
                    test_text = line.split(':', 1)[1].strip()
                    continue

                if name_text is None and line.startswith('name:'):
                    name_text = line.split(':', 1)[1].strip()
                    continue

                if (
                    rgb_eq is not None and thermal_eq is not None
                    and test_text is not None and name_text is not None
                ):
                    break
    except Exception:
        return None, None, None, None

    return rgb_eq, thermal_eq, test_text, name_text


def _expected_equalization_from_run_context(run_opts_path, run_opts_data):
    """Infer expected equalization from run_opts first, then results.yaml fallbacks."""
    expected = _expected_equalization_type(
        run_opts_data.get('rgb_eq', 'none'),
        run_opts_data.get('thermal_eq', 'none'),
    )
    source = 'run_opts.rgb_eq_thermal_eq'

    results_path = run_opts_path.parent / 'results.yaml'
    rgb_from_results, th_from_results, test_text, name_text = _extract_results_hints_fast(results_path)
    has_explicit_results_eq = rgb_from_results is not None or th_from_results is not None
    if has_explicit_results_eq:
        expected = _expected_equalization_type(rgb_from_results, th_from_results)
        source = 'results.rgb_equalization_thermal_equalization'
        return expected, source

    hinted = _equalization_type_from_text(test_text)
    if hinted:
        expected = hinted
        source = 'results.test'
        return expected, source

    hinted = _equalization_type_from_text(name_text)
    if hinted:
        expected = hinted
        source = 'results.name'
        return expected, source

    return expected, source


def _is_train_run(run_mode):
    if isinstance(run_mode, list):
        return any(str(item).lower() == 'train' for item in run_mode)
    return str(run_mode or '').lower() == 'train'


def _missing_val_batch_artifacts(run_dir):
    """Return booleans indicating missing validation preview artifacts."""
    has_pred = any(run_dir.glob('val_batch*_pred.jpg'))
    has_labels = any(run_dir.glob('val_batch*_labels.jpg'))
    return (not has_pred), (not has_labels)


def analyze_directory_structure(root, apply_changes=False, fix_eq_mismatches=False):
    """Analyze (and optionally fix) common run directory ordering anomalies."""
    root_path = Path(root)
    eq_roots = EQ_ROOTS

    run_opts_errors = []
    eq_mismatches = []
    nested_anomalies = []
    missing_training_artifacts = []
    unknown_dformats = []
    unknown_conditions = []
    pending_eq_moves = []
    moves_done = 0

    # 1) Check run_opts consistency vs containing equalization folder.
    run_opts_paths = [
        p for p in root_path.rglob('run_opts.yaml')
        if not _is_ignored_relative_path(root_path, p)
    ]
    for run_opts_path in run_opts_paths:
        rel_parts = run_opts_path.relative_to(root_path).parts
        if not rel_parts:
            continue
        eq_folder = rel_parts[0]
        if eq_folder not in eq_roots:
            continue

        data = parseYaml(run_opts_path) or {}
        if not isinstance(data, dict) or not data:
            continue
        if 'error' in data:
            run_opts_errors.append(str(run_opts_path))
            continue

        dformat = str(data.get('dformat', '') or '')
        if dformat.startswith('unknown_'):
            unknown_dformats.append((str(run_opts_path), dformat))

        clist = data.get('clist', [])
        condition = ''
        if isinstance(clist, list) and clist:
            condition = str(clist[0])
        elif clist is not None:
            condition = str(clist)
        if condition.startswith('unknown_condition'):
            unknown_conditions.append((str(run_opts_path), condition))

        if _is_train_run(data.get('run_mode', [])):
            miss_pred, miss_labels = _missing_val_batch_artifacts(run_opts_path.parent)
            if miss_pred or miss_labels:
                missing_training_artifacts.append((str(run_opts_path), miss_pred, miss_labels))

        # Fast path: use run_opts first and only parse results.yaml on suspicious cases.
        expected_fast = _expected_equalization_type(data.get('rgb_eq', 'none'), data.get('thermal_eq', 'none'))
        if expected_fast != eq_folder:
            expected, expected_source = _expected_equalization_from_run_context(run_opts_path, data)
            if expected != eq_folder:
                rel_run_opts = run_opts_path.relative_to(root_path)
                rel_run_dir = rel_run_opts.parent
                target_rel_run_dir = Path(expected, *rel_run_dir.parts[1:])
                eq_mismatches.append((
                    str(run_opts_path),
                    eq_folder,
                    expected,
                    expected_source,
                    str(target_rel_run_dir),
                ))

                if fix_eq_mismatches:
                    current_run_dir = run_opts_path.parent
                    target_run_dir = root_path / target_rel_run_dir
                    pending_eq_moves.append((current_run_dir, target_run_dir))

    # Apply mismatch moves after scanning to avoid mutating tree during rglob iteration.
    if fix_eq_mismatches and pending_eq_moves:
        for current_run_dir, target_run_dir in pending_eq_moves:
            if not current_run_dir.exists():
                log(
                    f"Skipping move {current_run_dir} -> {target_run_dir}: source no longer exists",
                    bcolors.WARNING,
                )
                continue
            if target_run_dir.exists():
                log(
                    f"Cannot move mismatch run {current_run_dir} -> {target_run_dir}: target already exists",
                    bcolors.WARNING,
                )
                continue
            target_run_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(current_run_dir), str(target_run_dir))
            moves_done += 1

    # 2) Detect/fix accidental nested equalization directories.
    for eq in eq_roots:
        eq_path = root_path / eq
        if not eq_path.exists():
            continue

        for nested in eq_path.rglob('variance_*_equalization'):
            # Expected anomaly pattern:
            # <eq>/<variance_group>/<variance_*_equalization>/<run_dir>
            parent = nested.parent
            if parent.parent != eq_path:
                continue
            if not parent.name.startswith('variance_'):
                continue

            run_dirs = [
                p for p in nested.iterdir()
                if p.is_dir() and (p / 'args.yaml').exists() and (p / 'results.yaml').exists()
            ]
            if not run_dirs:
                continue

            nested_anomalies.append((str(nested), len(run_dirs)))
            if apply_changes:
                for run_dir in run_dirs:
                    target = parent / run_dir.name
                    if target.exists():
                        log(f"Cannot move {run_dir} -> {target}: target already exists", bcolors.WARNING)
                        continue
                    shutil.move(str(run_dir), str(target))
                    moves_done += 1
                # Remove wrapper if it became empty.
                try:
                    if not any(nested.iterdir()):
                        nested.rmdir()
                except Exception:
                    pass

    # Report summary.
    log(
        f"Directory analysis: run_opts_errors={len(run_opts_errors)}, eq_mismatches={len(eq_mismatches)}, "
        f"nested_anomalies={len(nested_anomalies)}, missing_training_artifacts={len(missing_training_artifacts)}, moved_runs={moves_done}",
        bcolors.HEADER,
    )

    for path in run_opts_errors[:20]:
        log(f"run_opts with error: {path}", bcolors.WARNING)
    if len(run_opts_errors) > 20:
        log(f"... and {len(run_opts_errors) - 20} more run_opts with error", bcolors.WARNING)

    for path, current_eq, expected_eq, expected_source, target_rel in eq_mismatches[:20]:
        log(
            f"eq mismatch: {path} (folder={current_eq}, expected={expected_eq}, source={expected_source}, target={target_rel})",
            bcolors.WARNING,
        )
    if len(eq_mismatches) > 20:
        log(f"... and {len(eq_mismatches) - 20} more equalization mismatches", bcolors.WARNING)

    for nested_path, count in nested_anomalies[:20]:
        action = "fixed" if apply_changes else "detected"
        log(f"nested anomaly {action}: {nested_path} (runs={count})", bcolors.WARNING)
    if len(nested_anomalies) > 20:
        log(f"... and {len(nested_anomalies) - 20} more nested anomalies", bcolors.WARNING)

    for path, miss_pred, miss_labels in missing_training_artifacts[:20]:
        log(
            f"incomplete training artifacts: {path} (missing_pred={miss_pred}, missing_labels={miss_labels})",
            bcolors.WARNING,
        )
    if len(missing_training_artifacts) > 20:
        log(
            f"... and {len(missing_training_artifacts) - 20} more runs missing training artifacts",
            bcolors.WARNING,
        )

    report_csv_path = root_path / ANOMALY_REPORT_CSV
    report_txt_path = root_path / ANOMALY_REPORT_TXT

    with open(report_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["anomaly_type", "path", "current_value", "expected_value", "extra"])
        for path in run_opts_errors:
            writer.writerow(["run_opts_error", path, "error", "", ""])
        for path, current_eq, expected_eq, expected_source, target_rel in eq_mismatches:
            writer.writerow([
                "equalization_mismatch",
                path,
                current_eq,
                expected_eq,
                f"source={expected_source};target={target_rel}",
            ])
        for nested_path, count in nested_anomalies:
            writer.writerow(["nested_anomaly", nested_path, count, "", ""])
        for path, miss_pred, miss_labels in missing_training_artifacts:
            writer.writerow([
                "missing_training_artifacts",
                path,
                f"missing_pred={miss_pred},missing_labels={miss_labels}",
                "val_batch*_pred.jpg and val_batch*_labels.jpg present",
                "",
            ])
        for path, dformat in unknown_dformats:
            writer.writerow(["unknown_dformat", path, dformat, "", ""])
        for path, condition in unknown_conditions:
            writer.writerow(["unknown_condition", path, condition, "", ""])

    with open(report_txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write("Run opts anomaly report\n")
        txt_file.write(f"root: {root}\n\n")
        txt_file.write(
            f"Summary: run_opts_errors={len(run_opts_errors)}, eq_mismatches={len(eq_mismatches)}, "
            f"nested_anomalies={len(nested_anomalies)}, missing_training_artifacts={len(missing_training_artifacts)}, "
            f"unknown_dformats={len(unknown_dformats)}, "
            f"unknown_conditions={len(unknown_conditions)}, moved_runs={moves_done}\n\n"
        )

        txt_file.write("[run_opts_errors]\n")
        for path in run_opts_errors:
            txt_file.write(f"- {path}\n")

        txt_file.write("\n[equalization_mismatches]\n")
        for path, current_eq, expected_eq, expected_source, target_rel in eq_mismatches:
            txt_file.write(
                f"- {path} | folder={current_eq} | expected={expected_eq} | source={expected_source} | target={target_rel}\n"
            )

        txt_file.write("\n[nested_anomalies]\n")
        for nested_path, count in nested_anomalies:
            txt_file.write(f"- {nested_path} | runs={count}\n")

        txt_file.write("\n[missing_training_artifacts]\n")
        for path, miss_pred, miss_labels in missing_training_artifacts:
            txt_file.write(
                f"- {path} | missing_pred={miss_pred} | missing_labels={miss_labels}\n"
            )

        txt_file.write("\n[unknown_dformats]\n")
        for path, dformat in unknown_dformats:
            txt_file.write(f"- {path} | dformat={dformat}\n")

        txt_file.write("\n[unknown_conditions]\n")
        for path, condition in unknown_conditions:
            txt_file.write(f"- {path} | clist={condition}\n")

    log(f"Anomaly reports written: {report_csv_path} and {report_txt_path}", bcolors.OKGREEN)

def generate_run_opts_file(root):
    try:
        args_path = os.path.join(root, 'args.yaml')
        results_path = os.path.join(root, 'results.yaml')

        args_data = parseYaml(args_path) or {}
        results_data = parseYaml(results_path) or {}
        if not isinstance(args_data, dict):
            args_data = {}
        if not isinstance(results_data, dict):
            results_data = {}

        run_opts = {}
        # Gather relevant data from args_data
        run_opts['batch'] = args_data['batch']
        run_opts['cache'] = args_data['cache']
        run_opts['deterministic'] = args_data['deterministic']
        run_opts['resume'] = args_data['resume']
        run_opts['run_mode'] = [args_data['mode']]

        model_path = args_data['model']
        run_opts['mlist'] = [Path(model_path).name]

        option_executed = _infer_option_executed(args_data, results_data)
        run_opts['olist'] = [option_executed]

        path = args_data['name']
        run_opts['path_name'] = os.path.dirname(path)

        run_opts['dataset'] = args_data['data']
        run_opts['device'] = args_data['device']

        run_opts['rgb_eq'] = results_data.get('rgb_equalization', 'none')
        run_opts['thermal_eq'] = results_data.get('thermal_equalization', 'none')
        run_opts['dformat'] = _infer_dataset_tag(args_data, results_data)
        run_opts['pretrained'] = results_data.get('pretrained', bool(args_data.get('pretrained', False)))
        
        args_data_data = str(args_data.get('data', '') or '')
        condition = _infer_condition_from_text(args_data_data)
        if not condition:
            condition = _infer_condition_from_text(args_data.get('name', ''))

        if condition == 'day':
            run_opts['clist'] = ['day']
        elif condition == 'night':
            run_opts['clist'] = ['night']
        else:
            run_opts['clist'] = [f"unknown_condition args_data_name:{args_data.get('name', '')}"]

        run_opts['iterations'] = 1 # Each folder is one iteration, truncate it here :)

        dumpYaml(os.path.join(root, 'run_opts.yaml'), run_opts, 'w')
        # Log removed: doesn't work properly with ProcessPoolExecutor
        return True
    except Exception as e:
        # Return error info to be logged in main process
        dumpYaml(os.path.join(root, 'run_opts.yaml'), {f"error": f"{str(e)}"}, 'w')
        return f"Error: {root} - {e}. args_data: {(args_data is None or args_data == {}) and 'None or empty' or 'Loaded'}; results_data: {(results_data is None or results_data == {}) and 'None or empty' or 'Loaded'}"


def gather_test_info(root):
    root_path = Path(root)
    files = [
        p for p in root_path.rglob('run_opts.yaml')
        if not _is_ignored_relative_path(root_path, p)
        and not (p.parent / 'EEHA_GUI_IGNORE').exists()  # don't count runs marked to ignore
    ]

    log(f"Found {len(files)} run_opts.yaml files to analyze.", bcolors.HEADER)
    information = {}
    detailed_rows_numeric = []

    # Process files in parallel and collect path_names
    def get_path_name(file_path):
        try:
            data = parseYaml(file_path)
            if not isinstance(data, dict) or not data:
                raise ValueError("run_opts.yaml is empty or invalid")
            if "error" in data:
                raise ValueError(f"run_opts generation error: {data['error']}")

            clist = data.get("clist", [])
            if isinstance(clist, list) and clist:
                condition = clist[0]
            else:
                condition = clist

            path_name_val = data.get('path_name', '') or ''
            condition = _normalize_condition(condition, path_name_val, str(file_path))

            dataset = data.get("dformat", "unknown_unknown_split")
            dataset = _normalize_dataset_tag(
                dataset,
                data.get('dataset', ''),
                path_name_val,
                str(file_path),
            )

            # default fusion from olist (primary source)
            fusion = None
            try:
                olist = data.get('olist')
                if isinstance(olist, list) and len(olist) > 0:
                    fusion = olist[0]
                else:
                    fusion = olist
            except Exception:
                fusion = None

            # try to detect a more specific fusion token from the path_name basename
            try:
                base = Path(path_name_val).name
                tokens = [t for t in base.split('_') if t]
                if fusion and fusion in tokens:
                    fusion = _normalize_fusion_name(fusion, path_name_val, condition)
            except Exception:
                pass

            if not fusion:
                fusion = 'unknown_fusion'

            friendly_name = f"{dataset}_{fusion}_{condition}"
            
            equalization_type = _expected_equalization_type(
                data.get("rgb_eq", "none"),
                data.get("thermal_eq", "none"),
            )
            path_name_split = path_name_val.split('/') if path_name_val else []
            if path_name_split and path_name_split[0] in {
                'no_equalization', 'th_equalization', 'rgb_equalization', 'rgb_th_equalization'
            }:
                equalization_type = path_name_split[0]

            # re-validation tag: npzfix / cross_kaist / cross_llvip (TestValidateYolo writes result_tag);
            # plain trainings/vals have none -> 'train' (treated as in-domain).
            tag = data.get('result_tag') or 'train'

            test_name = str(file_path).replace(str(root_path) + '/', '').replace('/run_opts.yaml', '')
            test_name = "/".join(test_name.split('/')[:-1])
            test_name = _normalize_test_path(test_name)

            # model dataset = where the MODEL was trained (from the variance_* group in the path),
            # distinct from `dataset` which is the TEST set it was validated ON (dformat). For cross
            # evals these differ; for in-domain/train they coincide.
            model_dataset = ('kaist' if 'variance_kaist' in test_name else
                             ('llvip' if 'variance_llvip' in test_name else 'unknown'))
            # print(f"Final test_name: {test_name}")
            # return_data = data["path_name"], {"condition": condition, "fusion": fusion, "dataset": dataset, 'friendly_name': friendly_name}
            return_data = test_name, {"equalization_type": equalization_type,
                                      "condition": condition,
                                      "fusion": fusion,
                                      "dataset": dataset,
                                      'friendly_name': friendly_name,
                                      'tag': tag,
                                      'model_dataset': model_dataset,
                                      'test_path': test_name}
            return return_data
                   
        except Exception as e:
            log(f"Error reading {file_path}: {e}, file: {(data is None or data == {}) and 'None or empty' or 'Loaded'}", bcolors.ERROR)
            return None

    path_names = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_path_name, f): f for f in files}
        for future in as_completed(futures):
            data = future.result()
            if data:
                path_names.append(data)

    # Split complete tests from folders with subtests:
    test_columns_index = {}
    complete_tests = {}
    sub_folder_tests = {}

    for (test_name, content) in path_names:
        parts = Path(test_name).parts
        if len(parts) > 1:
            if parts[0] not in test_columns_index:
                test_columns_index[parts[0]] = len(list(test_columns_index.keys()))
            if test_name not in sub_folder_tests:
                sub_folder_tests[test_name] = {'count': 0, 'content': content}
            sub_folder_tests[test_name]['count'] += 1
        else:
            if test_name not in complete_tests:
                complete_tests[test_name] = {'count': 0, 'content': content}
            complete_tests[test_name]['count'] += 1

    # Reformat to group by test name with its different conditions inside
    sub_folder_tests_by_name = {}
    for test_name, test_info in sub_folder_tests.items():
        parts = Path(test_name).parts
        if len(parts) < 2:
            continue
        condition = parts[0]
        name = '/'.join(parts[1:])
        if name not in sub_folder_tests_by_name:
            sub_folder_tests_by_name[name] = {}
        sub_folder_tests_by_name[name][condition] = test_info

    ## Make tables:
    headers_general = ["Test name", "Count"]
    general_rows = []
    for key, test_info in complete_tests.items():
        general_rows.append([f"{test_info['content']['friendly_name']} ({key})", test_info['count']])

    general_rows = sorted(general_rows, key=lambda x: x[0])
    table_general = tabulate.tabulate(general_rows, headers=headers_general, tablefmt="fancy_grid", colalign = None, showindex=True)
    log(f"\nSingle Tests Summary:\n{table_general}\n", bcolors.OKCYAN)

    ## Now process subfolder tests
    headers = ["Test Name"] + list(test_columns_index.keys())
    rows = []
    for test_name, conditions in sub_folder_tests_by_name.items():
        row = [" "]*(len(test_columns_index)+1)
        for condition, test_info in conditions.items():
            index = test_columns_index[condition] + 1
            number_count = test_info['count']
            if number_count < 5:
                row[index] = f"{bcolors.ERROR}{test_info['count']}{bcolors.ENDC}{bcolors.OKCYAN}"
            else:
                row[index] = test_info['count']
            row[0] = test_info['content']['friendly_name'] + f" ({test_name})"
            #row[0] = test_name
        rows.append(row)
    
    rows = sorted(rows, key=lambda x: x[0])
    table = tabulate.tabulate(rows, headers=headers, tablefmt="fancy_grid", colalign = ("center", "left")+("center",)*len(test_columns_index), showindex=True)
    log(f"\nGrouped Tests Summary:\n{table}\n", bcolors.OKCYAN)

    # ---- Two matrices, split by MODEL dataset (where the model was TRAINED) -----------------------
    # Columns = fusion methods. Rows = (eq, test_dataset, condition) = what each model was validated WITH.
    # A row whose test dataset != the model's dataset is a CROSS row (marked "(cross)").
    # In-domain cells: npzfix (corrected re-val) predominates over the original train-time val.
    # cell[model_ds][(eq, test_ds, cond)][fusion][tag] = count
    cell = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
    methods_seen = set()
    for test_name, conditions in sub_folder_tests_by_name.items():
        for eq_condition, test_info in conditions.items():
            content = test_info.get('content', {})
            fusion = content.get('fusion', 'unknown_fusion')
            eq_type = content.get('equalization_type', 'unknown_equalization')
            test_ds = content.get('dataset', '')
            condition = content.get('condition', '')
            tag = content.get('tag', 'train')
            model_ds = content.get('model_dataset', 'unknown')
            count = test_info.get('count', 0)
            cell[model_ds][(eq_type, test_ds, condition)][fusion][tag] += count
            methods_seen.add(fusion)

    def _indomain(tc):
        # npzfix is the corrected re-val of the SAME trained models -> predominates over train.
        for t in ('npzfix', 'train', 'val'):
            if tc.get(t):
                return tc[t]
        return 0

    def _cross(tc):
        return tc.get('cross_kaist', 0) + tc.get('cross_llvip', 0)

    methods = sorted(methods_seen)
    empty_cell = f"{bcolors.ERROR}-{bcolors.ENDC}{bcolors.OKCYAN}"
    for model_ds in sorted(cell.keys()):
        rows = []
        for rk in sorted(cell[model_ds].keys()):
            eq_type, test_ds, condition = rk
            is_cross = not str(test_ds).startswith(model_ds)
            valfn = _cross if is_cross else _indomain
            cells = [valfn(cell[model_ds][rk].get(m, {})) for m in methods]
            ds_line = ("(cross) " if is_cross else "") + str(test_ds)
            label = f"{eq_type}\n{ds_line}\n{condition}"
            row = [label]
            for v in cells:
                if isinstance(v, int) and 0 < v < 5:
                    row.append(f"{bcolors.ERROR}{v}{bcolors.ENDC}{bcolors.OKCYAN}")
                elif v == 0:
                    row.append(empty_cell)
                else:
                    row.append(v)
            rows.append(row)
            kind = 'cross' if is_cross else 'in_domain'
            for m, v in zip(methods, cells):
                detailed_rows_numeric.append([model_ds, kind, eq_type, test_ds, condition, m, v])
        headers = [f"model={model_ds}\n(eq / test / cond)"] + methods
        n_cols = len(headers)
        colalign = ("center", "left") + ("center",) * (n_cols - 1)
        table = tabulate.tabulate(rows, headers=headers, tablefmt="fancy_grid", colalign=colalign, showindex=True)
        log(f"\nMatrix — modelos entrenados en {str(model_ds).upper()} "
            f"(fila = validado con; '(cross)' = dataset contrario; in-domain: npzfix>train):\n{table}\n",
            bcolors.OKCYAN)

    detailed_csv_path = root_path / DETAILED_COUNTS_CSV
    with open(detailed_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["model_dataset", "kind", "equalization_type", "test_dataset", "condition", "fusion", "count"])
        for row in sorted(detailed_rows_numeric, key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5])):
            writer.writerow(row)
    log(f"Detailed count report written: {detailed_csv_path}", bcolors.OKGREEN)

if __name__ == "__main__":
    # Collect all directories to process
    dirs_to_process = []
    already_existed = 0
    skipped_error_run_opts = 0
    already_existed_examples = []
    skipped_error_examples = []
    for root, dirs, files in os.walk(BASE_SEARCH_PATH):
        # Skip excluded top-level groups (for example: BASE_SEARCH_PATH/other).
        if root == BASE_SEARCH_PATH:
            dirs[:] = [d for d in dirs if d not in IGNORED_TOP_LEVEL_DIRS]

        # Skip excluded variance groups under equalization roots.
        try:
            rel_parts = Path(root).relative_to(Path(BASE_SEARCH_PATH)).parts
            if len(rel_parts) == 1 and rel_parts[0] in EQ_ROOTS:
                dirs[:] = [d for d in dirs if d not in IGNORED_VARIANCE_GROUP_DIRS]
        except Exception:
            pass

        if 'args.yaml' in files and 'results.yaml' in files:
            if 'run_opts.yaml' in files and not FORCE_REGENERATE:
                run_opts_path = os.path.join(root, 'run_opts.yaml')
                should_regenerate = False
                try:
                    existing_run_opts = parseYaml(run_opts_path) or {}
                    should_regenerate = isinstance(existing_run_opts, dict) and 'error' in existing_run_opts
                except Exception:
                    # If we cannot parse the file, regenerate it to self-heal.
                    should_regenerate = True

                if should_regenerate:
                    if REGENERATE_ERROR_RUN_OPTS:
                        log(f"run_opts.yaml in {root} contains error/invalid data, regenerating...", bcolors.WARNING)
                    else:
                        skipped_error_run_opts += 1
                        if len(skipped_error_examples) < 10:
                            skipped_error_examples.append(root)
                        continue
                else:
                    already_existed += 1
                    if len(already_existed_examples) < 10:
                        already_existed_examples.append(root)
                    continue
            dirs_to_process.append(root)

    if already_existed_examples:
        log(
            f"run_opts.yaml already exists in {already_existed} folders. Examples: {already_existed_examples}",
            bcolors.WARNING,
        )
    if skipped_error_examples:
        log(
            f"run_opts with error skipped in {skipped_error_run_opts} folders. Examples: {skipped_error_examples}",
            bcolors.WARNING,
        )

    # Parallelize the generation of run_opts.yaml with progress bar
    if dirs_to_process:
        log(f"{already_existed} already existed. Processing {len(dirs_to_process)} folders in parallel...", bcolors.HEADER)
        with ThreadedProcessPoolExecutor(max_processes=10, max_threads=10) as executor:
            log(f"Launching {executor._max_processes} processes in parallel with {executor._max_threads} threads each.")
            futures = [executor.submit(generate_run_opts_file, d) for d in dirs_to_process]
            errors = []
            with tqdm(total=len(futures), desc="Generating run_opts.yaml", unit="file") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result is not True:
                        errors.append(result)
                    pbar.update(1)
            
            if errors:
                log(f"\n{len(errors)} errors occurred:", bcolors.ERROR)
                for error in errors[:10]:  # Show first 10 errors
                    log(f"  {error}", bcolors.ERROR)
                if len(errors) > 10:
                    log(f"  ... and {len(errors) - 10} more errors", bcolors.ERROR)
    else:
        log("No folders to process.", bcolors.WARNING)

    log(
        f"Generated {len(dirs_to_process)} run_opts.yaml files. {already_existed} already generated before. "
        f"{skipped_error_run_opts} existing error run_opts skipped. Now gathering test info...",
        bcolors.HEADER,
    )

    if ANALYZE_DIRECTORY_STRUCTURE:
        analyze_directory_structure(
            BASE_SEARCH_PATH,
            apply_changes=FIX_DIRECTORY_STRUCTURE,
            fix_eq_mismatches=FIX_EQUALIZATION_MISMATCHES,
        )

    gather_test_info(BASE_SEARCH_PATH)