
from .dataset_manager import DataSetHandler
from .train_data_summary_tab import TrainDataSummary
from .train_compare_tab import TrainComparePlotter
from .train_eval_tab import TrainEvalPlotter
from .variance_compare_tab import VarianceComparePlotter
from .csv_table_tab import CSVTablePlotter

# Lazy import to avoid heavy dependencies from argument_parser
def __getattr__(name):
    if name == "SchedulerHandlerPlotter":
        from .scheduler_tab import SchedulerHandlerPlotter
        return SchedulerHandlerPlotter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")