from typing import Any
from datasets import Dataset
from kedro.io import AbstractDataset


class HFDataset(AbstractDataset):
    def __init__(self, filepath):
        self._filepath = filepath

    def _load(self) -> Dataset:
        return Dataset.load_from_disk(self._filepath)

    def _save(self, data: Dataset) -> None:
        data.save_to_disk(self._filepath)

    def _describe(self) -> dict[str, Any]:
        return {
            "file_path": self._filepath
        }
