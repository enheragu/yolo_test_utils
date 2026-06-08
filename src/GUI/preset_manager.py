#!/usr/bin/env python3
# encoding: utf-8
"""
Save and load GUI plot presets (selection + mode + export filters) as YAML.

A preset captures everything needed to reproduce a plot without manual selection:
- which datasets are checked (individual trials and variance groups)
- mode (best / p50 / p75 / p90 / p95) — may be a list for batch export
- detection class
- which plots and CSV columns to export

YAML schema (v1):

    version: 1
    tab: train_compare
    mode: p90                 # or [best, p90]
    class: all
    selection:
      datasets: [key1, key2, ...]
      variance_groups: [group1, group2, ...]
    export:
      plots:     [train_compare/PR Curve, train_compare/mAP50, ...]
                                          # tab-qualified: <tab_id>/<plot_name>
                                          # unqualified names are interpreted as the current tab
                                          # optional, [] or missing = all
      columns:       [Label, Model, mAP50, ...]  # CSV column subset
      table_formats: [csv, tex]                  # subset of {csv, txt, tex, html}
                                                 # [] = save no tables; missing = all four
      plot_formats:  [pdf]                       # subset of {png, pdf}
                                                 # [] = save no plots; missing = png only
      languages:     [es, en]                    # missing = use GUI config default
      filename:      outputs/paper_v1/kaist_day_methods
                                                 # optional base path (no extension);
                                                 # GUI pre-fills Save dialog; CLI uses directly
"""

from typing import Optional

from utils import parseYaml, dumpYaml, log, bcolors


PRESET_VERSION = 1


def build_preset(tab_id, mode, class_key, dataset_keys, variance_group_keys,
                 export_plots=None, export_columns=None,
                 export_table_formats=None, export_plot_formats=None,
                 export_languages=None, export_filename=None):
    """Build a preset dict from the current GUI state.

    export_plots: list of plot names (unqualified); will be saved as '<tab_id>/<name>'
                  to support multi-tab presets in the future.
    """
    preset = {
        'version': PRESET_VERSION,
        'tab':     tab_id,
        'class':   class_key or 'all',
        'selection': {
            'datasets':        sorted(dataset_keys),
            'variance_groups': sorted(variance_group_keys),
        },
        'export': {},
    }
    if mode is not None:
        preset['mode'] = mode
    if export_plots:
        preset['export']['plots'] = [f'{tab_id}/{p}' for p in export_plots]
    if export_columns:
        preset['export']['columns'] = list(export_columns)
    if export_table_formats is not None:
        preset['export']['table_formats'] = list(export_table_formats)
    if export_plot_formats is not None:
        preset['export']['plot_formats'] = list(export_plot_formats)
    if export_languages:
        preset['export']['languages'] = list(export_languages)
    if export_filename:
        preset['export']['filename'] = export_filename
    return preset


def plots_for_tab(plots_list, tab_id):
    """Filter export.plots to those for a given tab, stripping the prefix.

    Names without a '/' are kept as-is (legacy / unqualified).
    """
    if not plots_list:
        return None
    result = []
    for p in plots_list:
        if '/' in p:
            t, name = p.split('/', 1)
            if t == tab_id:
                result.append(name)
        else:
            result.append(p)
    return result or None


def save_preset(file_path, preset):
    """Write preset dict to YAML at file_path."""
    dumpYaml(file_path, preset)
    log(f"[preset] Saved preset to {file_path}")


def load_preset(file_path) -> Optional[dict]:
    """Load preset YAML, return dict or None on failure."""
    try:
        data = parseYaml(file_path)
    except Exception as e:
        log(f"[preset] Could not read {file_path}: {e}", bcolors.ERROR)
        return None

    if not isinstance(data, dict):
        log(f"[preset] Malformed preset (not a dict): {file_path}", bcolors.ERROR)
        return None

    version = data.get('version')
    if version != PRESET_VERSION:
        log(f"[preset] Unexpected preset version {version} (expected {PRESET_VERSION})", bcolors.WARNING)

    return data


def preset_modes(preset):
    """Return list of modes (always a list, even if YAML had a single string)."""
    m = preset.get('mode', 'best')
    return m if isinstance(m, list) else [m]
