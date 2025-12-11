# storm/plugins/base.py
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd

class ModelPlugin(ABC):
    key: str  # short name, e.g. "state", "perturbench"

    def prepare_dirs(self, layout) -> None:
        pass  # optional

    @abstractmethod
    def emit_for_split(
        self, df: pd.DataFrame, dataset_name: str, split_name: str,
        h5ad_path: Path,
        perturbation_key: str, covariate_key: str, control_value: str, seed: int, layout
    ) -> None:
        ...

    @abstractmethod
    def visualize_heatmap() -> None:
        pass

    @abstractmethod
    def visualize_scatterplots() -> None:
        pass