#!/usr/bin/env python3
# encoding: utf-8

## NEEDS PYQT 6: pip install PyQt6  
## Also needs to install lib: sudo apt install libxcb-cursor0
## might need an update -> pip install --upgrade pip

# If orphans are left behind just use:
# -> kill $(ps -aux | grep "gui_plot" | awk '{print $2}')

import sys
import os
import argparse
from datetime import datetime

from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QMenu, QFileDialog

import time as _time

from utils import log, bcolors
from GUI.gui_config import set_language, get_language, LANGUAGES, is_tab_enabled, set_export_mode
from GUI import DataSetHandler

sys.path.append('.')
import src # Imports __init__.py defined in paralel to this script

## Data YAML cached for faster data loading
# Update means that previous will be overwriten
update_cache = False
load_from_cache = False
lazy_load = True  # Lazy loading: load datasets on-demand instead of all at startup


# ── Tee stdout/stderr to a log file for debugging ──
class _TeeWriter:
    """Wraps a stream and mirrors writes to a log file."""

    def __init__(self, original, log_file):
        self._original = original
        self._log_file = log_file

    def write(self, msg):
        self._original.write(msg)
        try:
            self._log_file.write(msg)
            self._log_file.flush()
        except Exception:
            pass

    def flush(self):
        self._original.flush()

    def fileno(self):
        return self._original.fileno()

    # Delegate everything else (isatty, encoding, …) to original
    def __getattr__(self, name):
        return getattr(self._original, name)


def _setup_gui_log():
    """Create timestamped log file and tee stdout/stderr into it."""
    log_dir = os.path.join(os.getenv('HOME', '~'), '.cache', 'eeha_gui_cache', 'gui_log')
    os.makedirs(log_dir, exist_ok=True)

    # Prune old logs — keep only the 20 most recent
    try:
        logs = sorted(
            [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith('.log')],
            key=os.path.getmtime,
        )
        for stale in logs[:-20]:
            os.remove(stale)
    except Exception:
        pass

    timetag = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    log_path = os.path.join(log_dir, f'{timetag}_gui.log')
    log_file = open(log_path, 'a', encoding='utf-8')

    sys.stdout = _TeeWriter(sys.stdout, log_file)
    sys.stderr = _TeeWriter(sys.stderr, log_file)
    return log_path


class LazyTab(QWidget):
    """Placeholder widget that defers real tab creation until the user clicks on it."""
    def __init__(self, factory, dataset_handler):
        super().__init__()
        self._factory = factory          # callable: class to instantiate
        self._dataset_handler = dataset_handler
        self._real_widget = None
        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)

    def ensure_initialized(self) -> QWidget:
        """Build the real tab widget if not done yet. Returns it."""
        if self._real_widget is None:
            t0 = _time.time()
            name = self._factory.__name__
            log(f"[LazyTab] Initializing {name}...")
            self._real_widget = self._factory(self._dataset_handler)
            self._layout.addWidget(self._real_widget)
            log(f"[LazyTab] {name} ready in {_time.time() - t0:.2f}s.")
        return self._real_widget

    # Delegate public API used by update_view_and_menu
    def update_view_and_menu(self, menu_list):
        widget = self.ensure_initialized()
        widget.update_view_and_menu(menu_list)

    def update_checkbox(self):
        if self._real_widget:
            self._real_widget.update_checkbox()


class GUIPlotter(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Training result analyzer")
        self.setGeometry(100, 100, 1400, 1000)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)
        
        self.dataset_handler = DataSetHandler(update_cache, load_from_cache=load_from_cache, lazy_load=lazy_load)

        # Add tabs based on config/features.py settings
        # Tabs are wrapped in LazyTab so they are only built when the user clicks on them.
        # This avoids creating thousands of Qt widgets at startup.
        tab_defs = []
        if is_tab_enabled('train_compare'):
            from GUI import TrainComparePlotter
            tab_defs.append((TrainComparePlotter, "Compare training data"))
        if is_tab_enabled('variance_compare'):
            from GUI import VarianceComparePlotter
            tab_defs.append((VarianceComparePlotter, "Variance comparison"))
        if is_tab_enabled('train_eval'):
            from GUI import TrainEvalPlotter
            tab_defs.append((TrainEvalPlotter, "Review training process"))
        if is_tab_enabled('csv_table'):
            from GUI import CSVTablePlotter
            tab_defs.append((CSVTablePlotter, "Table"))
        
        for i, (factory, title) in enumerate(tab_defs):
            if i == 0:
                # Build the first tab immediately (it's visible at startup)
                t0 = _time.time()
                tab = factory(self.dataset_handler)
                log(f"[main] First tab ({factory.__name__}) built in {_time.time() - t0:.2f}s.")
            else:
                # Defer other tabs until the user clicks on them
                tab = LazyTab(factory, self.dataset_handler)
            self.tab_widget.addTab(tab, title)

        if is_tab_enabled('scheduler'):
            from GUI import SchedulerHandlerPlotter
            scheduler_tab = SchedulerHandlerPlotter()
            self.tab_widget.addTab(scheduler_tab, "Scheduled tests")
                
        self.tab_widget.currentChanged.connect(self.update_view_and_menu)
        self.update_view_and_menu()  # Actualizar el menú "View" cuando se abre la ventana

    def reload_cached_dataset(self):
        self.dataset_handler.new(load_from_cache = True)
    
    def reload_raw_dataset(self):
        self.dataset_handler.new(update_cache = True)

    def reload_incomplete_dataset(self):
        self.dataset_handler.reloadIncomplete()

    def reload_from_path(self):
        search_path = QFileDialog.getExistingDirectory(self, "Select path")
        if search_path:
            log(f"[main] Reloading data from path: {search_path}")
            self.dataset_handler.new(update_cache = True, search_path = search_path)

    def _set_export_language(self, lang):
        set_language(lang)
        log(f"[main] Export language set to: {lang}")
        

    def update_view_and_menu(self):
        # Limpiar el menú "View"
        self.menuBar().clear()
        archive_menu = self.menuBar().addMenu('Archive')

        self.reload_datasets_menu = QMenu('Reload datasets')
        
        self.use_cached_datasets_action = QAction('Reload cached datasets')
        self.use_cached_datasets_action.triggered.connect(self.reload_cached_dataset)
        self.reload_datasets_menu.addAction(self.use_cached_datasets_action)
        
        self.reload_from_raw_action = QAction('Reload all from raw')
        self.reload_from_raw_action.triggered.connect(self.reload_raw_dataset)
        self.reload_datasets_menu.addAction(self.reload_from_raw_action)

        self.reload_incomplete_action = QAction('Reload incomplete from raw')
        self.reload_incomplete_action.triggered.connect(self.reload_incomplete_dataset)
        self.reload_datasets_menu.addAction(self.reload_incomplete_action)

        archive_menu.addMenu(self.reload_datasets_menu)

        self.reload_from_path_menu = QAction('Load datasets from path')
        self.reload_from_path_menu.triggered.connect(self.reload_from_path )
        archive_menu.addAction(self.reload_from_path_menu)

        view_menu = self.menuBar().addMenu('View')
        tools_menu = self.menuBar().addMenu('Tools')
        edit_menu = self.menuBar().addMenu('Edit')

        # Language menu for export
        language_menu = QMenu('Language / Idioma', self)
        self.language_action_group = QActionGroup(self)
        self.language_action_group.setExclusive(True)
        
        lang_names = {'en': 'English', 'es': 'Español'}
        for lang in LANGUAGES:
            action = QAction(lang_names.get(lang, lang), self, checkable=True)
            action.setData(lang)
            action.setChecked(lang == get_language())
            action.triggered.connect(lambda checked, l=lang: self._set_export_language(l))
            self.language_action_group.addAction(action)
            language_menu.addAction(action)
        
        view_menu.addMenu(language_menu)

        # Obtener la pestaña/tab actual
        current_tab_widget = self.tab_widget.currentWidget()
        current_tab_widget.update_view_and_menu([archive_menu, view_menu, tools_menu, edit_menu])
        

def handleArguments():
    global update_cache, load_from_cache, lazy_load
    parser = argparse.ArgumentParser(description="GUI to review training results.")
    parser.add_argument('--force_update_cache', action='store_true', help='Forces an update of all cache data from the data folder from scratch.')
    parser.add_argument('--load_from_cache', action='store_true', help='Use cached data without checking the complete data.')
    parser.add_argument('--no_lazy_load', action='store_true', help='Disable lazy loading - load all datasets at startup (legacy behavior).')
    parser.add_argument('--export_lang', type=str, default='en', choices=LANGUAGES + ['all'], 
                        help=f'Language for exported plots/tables. Use "all" to export in all languages. Options: {LANGUAGES + ["all"]}. Default: en')

    args = parser.parse_args()

    if args.force_update_cache and args.load_from_cache:
        parser.error('--force_update_cache and --load_from_cache are mutually exclusive')

    update_cache = args.force_update_cache
    load_from_cache = args.load_from_cache
    lazy_load = not args.no_lazy_load
    
    # Set export mode and initial language
    set_export_mode(args.export_lang)
    initial_lang = 'en' if args.export_lang == 'all' else args.export_lang
    set_language(initial_lang)
    log(f"[main] Update cache: {update_cache}, Load from cache: {load_from_cache}, Lazy load: {lazy_load}, Export language: {args.export_lang}.")

def main():
    app = QApplication(sys.argv)
    font = QApplication.font()
    font.setPointSize(10)  
    app.setFont(font)

    handleArguments()
    window = GUIPlotter()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    log_path = _setup_gui_log()
    log(f"[main] Start graphical interface for training result analysis.")
    log(f"[main] GUI log file: {log_path}")
    main()