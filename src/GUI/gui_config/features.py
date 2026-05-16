#!/usr/bin/env python3
# encoding: utf-8
"""
    Feature flags and configuration for the GUI.
    Toggle features on/off to control what gets loaded, cached, and displayed.
    
    Set to True to enable, False to disable.
"""

# =============================================================================
# MAIN TABS - Control which main tabs are shown in the GUI
# =============================================================================
TABS = {
    'train_compare': True,      # Compare training data tab
    'variance_compare': True,   # Variance comparison tab
    'train_eval': True,         # Review training process tab (evolution plots)
    'csv_table': True,          # Table export tab
    'scheduler': False,         # Scheduled tests tab (has heavy dependencies)
}

# =============================================================================
# TRAIN COMPARE TAB - Curve and metric plots
# =============================================================================
TRAIN_COMPARE_CURVES = {
    # Curve plots (line plots with confidence/recall on X axis)
    'PR Curve': True,
    'P Curve': True,
    'R Curve': True,
    'F1 Curve': True,
    'MR Curve': False,          # Miss Rate curve - disabled
    'MRFPPI Curve': False,      # MR-FPPI curve - disabled
}

TRAIN_COMPARE_METRICS = {
    # Bar/point plots (single value per dataset)
    'mAP50': True,
    'mAP50-95': True,
    'precision': True,
    'recall': True,
    'F1': True,
    'MissRate': False,          # Disabled
    'FPPI': False,              # Disabled
    'LAMR': False,              # Disabled
}

# =============================================================================
# VARIANCE COMPARE TAB - Statistical comparison plots
# =============================================================================
VARIANCE_COMPARE_PLOTS = {
    'PR Curve': False,          # Disabled for variance
    'P Curve': False,
    'R Curve': False,
    'F1 Curve': False,
    'MR Curve': False,
    'mAP50': True,
    'mAP50-95': True,
    'P': True,                  # Precision metric
    'R': True,                  # Recall metric
    'MR': False,
    'FPPI': False,
    'LAMR': False,
}

# =============================================================================
# TRAIN EVAL TAB - Training evolution plots
# =============================================================================
TRAIN_EVAL_PLOTS = {
    'Train Loss Ev.': True,     # Training loss evolution
    'Val Loss Ev.': True,       # Validation loss evolution
    'PR Evolution': True,       # Precision/Recall evolution
    'mAP Evolution': True,      # mAP evolution
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_train_compare_tab_keys():
    """Get active tab keys for TrainComparePlotter."""
    curves = [k for k, v in TRAIN_COMPARE_CURVES.items() if v]
    metrics = [k for k, v in TRAIN_COMPARE_METRICS.items() if v]
    return curves + metrics


def get_variance_compare_tab_keys():
    """Get active tab keys for VarianceComparePlotter."""
    return [k for k, v in VARIANCE_COMPARE_PLOTS.items() if v]


def get_train_eval_tab_keys():
    """Get active tab keys for TrainEvalPlotter."""
    return [k for k, v in TRAIN_EVAL_PLOTS.items() if v]


def is_tab_enabled(tab_name: str) -> bool:
    """Check if a main tab is enabled."""
    return TABS.get(tab_name, False)


def should_cache_data(data_type: str) -> bool:
    """
    Check if a data type should be computed/cached.
    Automatically deduced from enabled plots - no manual configuration needed.
    """
    # Gather all enabled plot keys from all tabs
    all_enabled = set()
    if TABS.get('train_compare', False):
        all_enabled.update(k for k, v in TRAIN_COMPARE_CURVES.items() if v)
        all_enabled.update(k for k, v in TRAIN_COMPARE_METRICS.items() if v)
    if TABS.get('variance_compare', False):
        all_enabled.update(k for k, v in VARIANCE_COMPARE_PLOTS.items() if v)
    
    # Map data types to the plots that need them
    data_dependencies = {
        'mr_plot': ['MR Curve', 'MR', 'MissRate', 'MRFPPI Curve', 'LAMR'],
        'fppi_plot': ['FPPI', 'MRFPPI Curve', 'LAMR'],
        'lamr': ['LAMR'],
        'csv_data': True,  # Always needed for tables if csv_table tab is enabled
        'confusion_matrix': False,  # Explicitly disabled - too large, rarely used
    }
    
    deps = data_dependencies.get(data_type)
    
    # Special cases
    if deps is True:
        return TABS.get('csv_table', False)
    if deps is False:
        return False
    if deps is None:
        return True  # Unknown data types are cached by default
    
    # Check if any plot that needs this data is enabled
    return any(plot in all_enabled for plot in deps)
