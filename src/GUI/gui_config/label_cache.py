#!/usr/bin/env python3
# encoding: utf-8
"""
    Cache for GUI settings between sessions.
    Stores label mappings and column selections in YAML files.
"""
import os
import yaml
from typing import Optional

# Cache file locations
_cache_dir = os.path.join(os.getenv('HOME', '~'), '.cache', 'eeha_gui_cache')
_label_cache_file = os.path.join(_cache_dir, 'label_mappings.yaml')
_column_cache_file = os.path.join(_cache_dir, 'column_selection.yaml')

# In-memory caches
_label_mappings: dict = {}
_label_loaded = False

_column_selection: Optional[list] = None
_column_loaded = False


def _ensure_cache_dir():
    """Ensure cache directory exists."""
    os.makedirs(_cache_dir, exist_ok=True)


def load_label_mappings() -> dict:
    """
    Load label mappings from cache file.
    Returns dict of {original: {lang: display, ...}}.
    Migrates old flat format ({original: display}) on load.
    """
    global _label_mappings, _label_loaded
    
    if _label_loaded:
        return _label_mappings.copy()
    
    if os.path.exists(_label_cache_file):
        try:
            with open(_label_cache_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    _label_mappings = _migrate_label_mappings(data)
        except Exception as e:
            print(f"Warning: Could not load label cache: {e}")
            _label_mappings = {}
    
    _label_loaded = True
    return _label_mappings.copy()


def _migrate_label_mappings(data: dict) -> dict:
    """
    Migrate old flat format {orig: display_str} to
    per-language format {orig: {en: display, es: display}}.
    Already-migrated entries (value is dict) are kept as-is.
    """
    migrated = {}
    for k, v in data.items():
        if isinstance(v, dict):
            migrated[k] = v  # already per-language
        elif isinstance(v, str):
            # Old format: apply same rename to all languages
            migrated[k] = {lang: v for lang in ('en', 'es')}
        # else skip invalid entries
    return migrated


def save_label_mappings(mappings: dict):
    """
    Save label mappings to cache file.
    Expects {original: {lang: display, ...}}.
    Only saves non-identity mappings.
    """
    global _label_mappings
    
    _ensure_cache_dir()
    
    # Update in-memory cache
    _label_mappings.update(mappings)
    
    # Remove identity mappings (where ALL language values equal the key)
    _label_mappings = {
        k: v for k, v in _label_mappings.items()
        if isinstance(v, dict) and any(lang_v != k for lang_v in v.values())
    }
    
    try:
        with open(_label_cache_file, 'w', encoding='utf-8') as f:
            yaml.dump(_label_mappings, f, allow_unicode=True, default_flow_style=False)
    except Exception as e:
        print(f"Warning: Could not save label cache: {e}")


def get_label_mappings() -> dict:
    """Get current label mappings (loads from file if not already loaded)."""
    return load_label_mappings()


def clear_label_cache():
    """Clear all stored label mappings."""
    global _label_mappings, _label_loaded
    
    _label_mappings = {}
    _label_loaded = True
    
    if os.path.exists(_label_cache_file):
        try:
            os.remove(_label_cache_file)
        except Exception as e:
            print(f"Warning: Could not delete label cache file: {e}")


# ============================================================================
# Column Selection Cache
# ============================================================================

def load_column_selection() -> Optional[list]:
    """
    Load column selection from cache file.
    Returns None if no selection is stored (meaning export all columns).
    """
    global _column_selection, _column_loaded
    
    if _column_loaded:
        return _column_selection.copy() if _column_selection else None
    
    if os.path.exists(_column_cache_file):
        try:
            with open(_column_cache_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if isinstance(data, list):
                    _column_selection = data
        except Exception as e:
            print(f"Warning: Could not load column cache: {e}")
            _column_selection = None
    
    _column_loaded = True
    return _column_selection.copy() if _column_selection else None


def save_column_selection(columns: Optional[list]):
    """
    Save column selection to cache file.
    Pass None to clear the selection (export all columns).
    """
    global _column_selection
    
    _ensure_cache_dir()
    
    _column_selection = columns.copy() if columns else None
    
    if _column_selection:
        try:
            with open(_column_cache_file, 'w', encoding='utf-8') as f:
                yaml.dump(_column_selection, f, allow_unicode=True, default_flow_style=False)
        except Exception as e:
            print(f"Warning: Could not save column cache: {e}")
    elif os.path.exists(_column_cache_file):
        # Remove file if no selection
        try:
            os.remove(_column_cache_file)
        except Exception:
            pass


def get_column_selection() -> Optional[list]:
    """Get current column selection (loads from file if not already loaded)."""
    return load_column_selection()


def clear_column_cache():
    """Clear stored column selection."""
    global _column_selection, _column_loaded
    
    _column_selection = None
    _column_loaded = True
    
    if os.path.exists(_column_cache_file):
        try:
            os.remove(_column_cache_file)
        except Exception as e:
            print(f"Warning: Could not delete column cache file: {e}")


# ============================================================================
# Plot Selection Cache  (per-tab: { tab_id: [selected_plot_keys] })
# ============================================================================

_plot_cache_file = os.path.join(_cache_dir, 'plot_selection.yaml')
_plot_selection: Optional[dict] = None
_plot_loaded = False


def load_plot_selection(tab_id: str) -> Optional[list]:
    """
    Load plot selection for a specific tab.
    Returns None if no selection stored (meaning export all plots).
    """
    global _plot_selection, _plot_loaded

    if not _plot_loaded:
        if os.path.exists(_plot_cache_file):
            try:
                with open(_plot_cache_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, dict):
                        _plot_selection = data
            except Exception as e:
                print(f"Warning: Could not load plot selection cache: {e}")
                _plot_selection = None
        _plot_loaded = True

    if _plot_selection and tab_id in _plot_selection:
        sel = _plot_selection[tab_id]
        return sel.copy() if isinstance(sel, list) else None
    return None


def save_plot_selection(tab_id: str, plots: Optional[list]):
    """
    Save plot selection for a specific tab.
    Pass None to clear the selection (export all plots).
    """
    global _plot_selection

    _ensure_cache_dir()

    if _plot_selection is None:
        _plot_selection = {}

    if plots is not None:
        _plot_selection[tab_id] = plots.copy()
    elif tab_id in _plot_selection:
        del _plot_selection[tab_id]

    if _plot_selection:
        try:
            with open(_plot_cache_file, 'w', encoding='utf-8') as f:
                yaml.dump(_plot_selection, f, allow_unicode=True, default_flow_style=False)
        except Exception as e:
            print(f"Warning: Could not save plot selection cache: {e}")
    elif os.path.exists(_plot_cache_file):
        try:
            os.remove(_plot_cache_file)
        except Exception:
            pass


def get_plot_selection(tab_id: str) -> Optional[list]:
    """Get current plot selection for a tab (loads from file if not already loaded)."""
    return load_plot_selection(tab_id)


def clear_plot_cache():
    """Clear all stored plot selections."""
    global _plot_selection, _plot_loaded

    _plot_selection = None
    _plot_loaded = True

    if os.path.exists(_plot_cache_file):
        try:
            os.remove(_plot_cache_file)
        except Exception as e:
            print(f"Warning: Could not delete plot selection cache file: {e}")
