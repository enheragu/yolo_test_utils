# GUI Config package - re-exports from main config for convenience
# This allows GUI modules to use `from GUI.config import ...` or `from .config import ...`
# The actual configuration files are in src/config/ to avoid circular imports

# GUI-specific config (features and i18n)
from .i18n import (
    t, t_lang, translate_list, set_language, get_language, get_plot_labels, LANGUAGES,
    set_export_mode, get_export_mode, get_export_languages, get_language_suffix
)
from .features import (
    TABS, TRAIN_COMPARE_CURVES, TRAIN_COMPARE_METRICS, VARIANCE_COMPARE_PLOTS, TRAIN_EVAL_PLOTS,
    get_train_compare_tab_keys, get_variance_compare_tab_keys, get_train_eval_tab_keys,
    is_tab_enabled, should_cache_data
)
from .label_cache import (
    load_label_mappings, save_label_mappings, get_label_mappings, clear_label_cache,
    load_column_selection, save_column_selection, get_column_selection, clear_column_cache,
    load_plot_selection, save_plot_selection, get_plot_selection, clear_plot_cache
)
