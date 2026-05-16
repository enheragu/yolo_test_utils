#!/usr/bin/env python3
# encoding: utf-8
"""
    Internationalization (i18n) support for GUI labels.
    Simple translation system for plot labels and UI elements.
"""
from typing import Optional

# Available languages
LANGUAGES = ['en', 'es']
_current_language = 'en'
_export_mode = 'en'  # 'en', 'es', or 'all'

# Translation dictionary: key -> {lang: translation}
_translations = {
    # Axis labels
    'Precision': {'en': 'Precision', 'es': 'Precisión'},
    'Recall': {'en': 'Recall', 'es': 'Recall'},
    'Confidence': {'en': 'Confidence', 'es': 'Confianza'},
    'Miss Rate': {'en': 'Miss Rate', 'es': 'Tasa de Fallos'},
    'Probability': {'en': 'Probability', 'es': 'Probabilidad'},
    
    # Plot titles / Tab names
    'PR Curve': {'en': 'PR Curve', 'es': 'Curva PR'},
    'P Curve': {'en': 'P Curve', 'es': 'Curva P'},
    'R Curve': {'en': 'R Curve', 'es': 'Curva R'},
    'F1 Curve': {'en': 'F1 Curve', 'es': 'Curva F1'},
    'MR Curve': {'en': 'MR Curve', 'es': 'Curva MR'},
    'MRFPPI Curve': {'en': 'MRFPPI Curve', 'es': 'Curva MR-FPPI'},
    
    # Metrics
    'precision': {'en': 'Precision', 'es': 'Precisión'},
    'recall': {'en': 'Recall', 'es': 'Recall'},
    
    # UI Elements
    'Class:': {'en': 'Class:', 'es': 'Clase:'},
    'Plot All': {'en': 'Plot All', 'es': 'Graficar Todo'},
    'Select All': {'en': 'Select All', 'es': 'Seleccionar Todo'},
    'Deselect All': {'en': 'Deselect All', 'es': 'Deseleccionar Todo'},
    'Generate Plot': {'en': 'Generate Plot', 'es': 'Generar Gráfico'},
    'Save Output': {'en': 'Save Output', 'es': 'Guardar Salida'},
    'Select': {'en': 'Select', 'es': 'Seleccionar'},
    
    # Menu items
    'Archive': {'en': 'Archive', 'es': 'Archivo'},
    'View': {'en': 'View', 'es': 'Vista'},
    'Tools': {'en': 'Tools', 'es': 'Herramientas'},
    'Edit': {'en': 'Edit', 'es': 'Editar'},
    'Language': {'en': 'Language', 'es': 'Idioma'},
    'English': {'en': 'English', 'es': 'Inglés'},
    'Spanish': {'en': 'Spanish', 'es': 'Español'},
    
    # Variance tab
    'Variance analysis sets:': {'en': 'Variance analysis sets:', 'es': 'Conjuntos de análisis de varianza:'},
    'Test groups:': {'en': 'Test groups:', 'es': 'Grupos de prueba:'},
    
    # Other
    'Model:': {'en': 'Model:', 'es': 'Modelo:'},
    'epoch': {'en': 'epoch', 'es': 'época'},
    
    # CSV Table Headers
    'Model': {'en': 'Model', 'es': 'Modelo'},
    'Condition': {'en': 'Condition', 'es': 'Condición'},
    'Type': {'en': 'Type', 'es': 'Tipo'},
    'Class': {'en': 'Class', 'es': 'Clase'},
    'Dataset': {'en': 'Dataset', 'es': 'Conjunto de datos'},
    'Best epoch (index)': {'en': 'Best epoch (index)', 'es': 'Mejor época (índice)'},
    'Train Duration (h)': {'en': 'Train Duration (h)', 'es': 'Duración entrenamiento (h)'},
    'Pretrained': {'en': 'Pretrained', 'es': 'Preentrenado'},
    'Deterministic': {'en': 'Deterministic', 'es': 'Determinístico'},
    'Batch Size': {'en': 'Batch Size', 'es': 'Tamaño de lote'},
    'Train Img': {'en': 'Train Img', 'es': 'Imágenes entrenamiento'},
    'Val Img': {'en': 'Val Img', 'es': 'Imágenes validación'},
    'Instances': {'en': 'Instances', 'es': 'Instancias'},
    'Num Classes': {'en': 'Num Classes', 'es': 'Núm. Clases'},
    'Device': {'en': 'Device', 'es': 'Dispositivo'},
    'Date': {'en': 'Date', 'es': 'Fecha'},
    'Title': {'en': 'Title', 'es': 'Título'},
    'Label': {'en': 'Label', 'es': 'Nombre'},
    'Group Key': {'en': 'Group Key', 'es': 'Clave de grupo'},
    'Table': {'en': 'Table', 'es': 'Tabla'},
}


def set_language(lang: str):
    """Set the current language for translations."""
    global _current_language
    if lang in LANGUAGES:
        _current_language = lang
    else:
        raise ValueError(f"Language '{lang}' not supported. Available: {LANGUAGES}")


def get_language() -> str:
    """Get the current language."""
    return _current_language


def set_export_mode(mode: str):
    """
    Set the export mode for multi-language export.
    
    Args:
        mode: 'en', 'es', or 'all' for exporting in all languages
    """
    global _export_mode
    if mode in LANGUAGES or mode == 'all':
        _export_mode = mode
    else:
        raise ValueError(f"Export mode '{mode}' not supported. Options: {LANGUAGES + ['all']}")


def get_export_mode() -> str:
    """Get the current export mode."""
    return _export_mode


def get_export_languages() -> list:
    """
    Get list of languages to export based on export mode.
    
    Returns:
        List of language codes, e.g. ['en'] or ['en', 'es']
    """
    if _export_mode == 'all':
        return LANGUAGES.copy()
    return [_export_mode]


def get_language_suffix(lang: Optional[str] = None) -> str:
    """
    Get language suffix for file names.
    Returns empty string for single-language mode, '_es' or '_en' for 'all' mode.
    """
    if _export_mode != 'all':
        return ''
    return f'_{lang or _current_language}'


def t(key: str) -> str:
    """
    Translate a key to the current language.
    Returns the key itself if no translation is found.
    """
    if key in _translations:
        return _translations[key].get(_current_language, key)
    return key


def t_lang(key: str, lang: str) -> str:
    """
    Translate a key to a specific language.
    Returns the key itself if no translation is found.
    """
    if key in _translations:
        return _translations[key].get(lang, key)
    return key


def translate_list(items: list, lang: Optional[str] = None) -> list:
    """
    Translate a list of strings to the specified language.
    
    Args:
        items: List of strings to translate
        lang: Target language. If None, uses current language.
    
    Returns:
        List of translated strings
    """
    use_lang = lang or _current_language
    return [t_lang(item, use_lang) for item in items]


def add_translation(key: str, translations: dict):
    """
    Add a new translation entry.
    
    Args:
        key: The translation key
        translations: Dict with language codes as keys, e.g. {'en': 'Hello', 'es': 'Hola'}
    """
    _translations[key] = translations


def get_plot_labels(lang: Optional[str] = None) -> dict:
    """
    Get plot label translations for export.
    Useful when you want labels in a specific language regardless of current UI language.
    
    Args:
        lang: Language code ('en' or 'es'). If None, uses current language.
    
    Returns:
        Dict with translated axis labels for plots.
    """
    use_lang = lang or _current_language
    
    return {
        'xlabel': {
            'Recall': _translations['Recall'][use_lang],
            'Confidence': _translations['Confidence'][use_lang],
            'FPPI': 'FPPI',  # Acronyms stay the same
            '': '',
        },
        'ylabel': {
            'Precision': _translations['Precision'][use_lang],
            'Recall': _translations['Recall'][use_lang],
            'F1': 'F1',
            'Miss Rate': _translations['Miss Rate'][use_lang],
            'MissRate': _translations['Miss Rate'][use_lang],
            'mAP50': 'mAP50',
            'mAP50-95': 'mAP50-95',
            'FPPI': 'FPPI',
            'LAMR': 'LAMR',
        }
    }
