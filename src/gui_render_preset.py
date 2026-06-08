#!/usr/bin/env python3
# encoding: utf-8
"""
Headless CLI to render plots and tables from a YAML preset.

Reuses the existing GUI tab classes via Qt's offscreen platform (no window opens),
so plot generation logic is shared with the interactive GUI exactly.

Usage:
    eeha_render_preset --preset gui_presets/.../foo.yaml
    eeha_render_preset --preset foo.yaml --output outputs/run42  # override filename base
"""

# ── MUST set offscreen BEFORE PyQt is imported ────────────────────────────
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys
import argparse
import time

from PyQt6.QtWidgets import QApplication

sys.path.append('.')
import src  # noqa: F401  — initializes path entries like gui_plot_results does

from utils import log, bcolors
from GUI.gui_config import set_export_mode, set_language
from GUI.dataset_manager import DataSetHandler
from GUI.train_compare_tab import TrainComparePlotter
from GUI.variance_compare_tab import VarianceComparePlotter
from GUI.preset_manager import load_preset


TAB_CLASSES = {
    'train_compare':    TrainComparePlotter,
    'variance_compare': VarianceComparePlotter,
}


def _resolve_modes(preset):
    """Return a list of (mode, suffix) tuples to iterate over."""
    mode = preset.get('mode')
    if mode is None:
        return [(None, '')]                       # tab without mode concept
    if isinstance(mode, list):
        return [(m, f'_{m}') for m in mode] or [(None, '')]
    return [(mode, f'_{mode}')]


def _apply_export_language(preset):
    """Set the global export mode so figures pick up the right languages."""
    langs = (preset.get('export') or {}).get('languages')
    if not langs:
        return
    if len(langs) > 1:
        set_export_mode('all')
    else:
        set_export_mode(langs[0])
        set_language(langs[0])


def render(preset_path, output_override=None, load_from_cache=False, force_update_cache=False):
    preset = load_preset(preset_path)
    if preset is None:
        log(f"[render] Could not load preset {preset_path}", bcolors.ERROR)
        return 1

    tab_id = preset.get('tab')
    if tab_id not in TAB_CLASSES:
        log(f"[render] Unsupported tab '{tab_id}'. Available: {list(TAB_CLASSES)}", bcolors.ERROR)
        return 1

    yaml_filename = (preset.get('export') or {}).get('filename')
    if output_override:
        out = os.path.expanduser(os.path.expandvars(output_override))
        # If --output is a directory (trailing slash or existing dir) and the YAML
        # provided a filename, combine them: <out>/<yaml basename>.
        # Otherwise treat --output as a full base path replacing the YAML filename.
        is_dir = out.endswith(os.sep) or out.endswith('/') or os.path.isdir(out)
        if is_dir and yaml_filename:
            base_path = os.path.join(out, os.path.basename(yaml_filename))
        else:
            base_path = out
    else:
        base_path = yaml_filename

    if not base_path:
        log("[render] Preset has no export.filename and no --output given", bcolors.ERROR)
        return 1

    log(f"[render] Output base path: {base_path}")

    _apply_export_language(preset)

    # load_from_cache=True scans ~/.cache/eeha_gui_cache (fast).
    # force_update_cache=True wipes and regenerates the cache from raw data.
    # lazy_load defers actual data reads until __getitem__ is called.
    log(f"[render] Loading DataSetHandler (load_from_cache={load_from_cache}, force_update_cache={force_update_cache}, lazy, no background)…")
    t0 = time.time()
    handler = DataSetHandler(
        update_cache=force_update_cache,
        load_from_cache=load_from_cache,
        lazy_load=True,
        start_background=False,  # CLI only loads what render_data() actually requests
    )
    log(f"[render] Handler ready in {time.time() - t0:.2f}s ({len(handler)} datasets discovered)")

    tab_cls = TAB_CLASSES[tab_id]
    tab = tab_cls(handler)
    tab._apply_preset_state(preset)

    iterations = _resolve_modes(preset)
    log(f"[render] Rendering {len(iterations)} variant(s): {[m for m, _ in iterations]}")

    for mode, suffix in iterations:
        if mode is not None and hasattr(tab, 'percentile_combobox'):
            idx = tab.percentile_combobox.findText(mode)
            if idx >= 0:
                tab.percentile_combobox.setCurrentIndex(idx)
            else:
                log(f"[render] Mode '{mode}' not in combobox; skipping", bcolors.WARNING)
                continue

        log(f"[render] === mode={mode or '(none)'} ===")
        t0 = time.time()
        tab.render_data()
        log(f"[render] Rendered in {time.time() - t0:.2f}s")

        out = f"{base_path}{suffix}"
        log(f"[render] Saving to {out}*")
        tab.save_outputs_to(out)

    log(f"[render] Done.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preset', required=True, help='Path to a preset YAML file')
    parser.add_argument('--output', default=None, help='Override the preset\'s export.filename base path')
    parser.add_argument('--load_from_cache', action='store_true',
                        help='Use cached data without checking the complete data (same as GUI flag).')
    parser.add_argument('--force_update_cache', action='store_true',
                        help='Forces an update of all cache data from the data folder from scratch (same as GUI flag).')
    args = parser.parse_args()

    if args.force_update_cache and args.load_from_cache:
        parser.error('--force_update_cache and --load_from_cache are mutually exclusive')

    # Qt application is needed even in offscreen mode (widgets need an event loop)
    _ = QApplication.instance() or QApplication(sys.argv)

    code = render(
        args.preset,
        args.output,
        load_from_cache=args.load_from_cache,
        force_update_cache=args.force_update_cache,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
