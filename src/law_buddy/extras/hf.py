from typing import Any
from datasets import Dataset, load_dataset
from kedro.io import AbstractDataset
from huggingface_hub import HfApi
import logging


logger = logging.getLogger(__file__)


class HFDataset(AbstractDataset):
    def __init__(self, filepath: str = None, dataset_name: str = None):
        """ Given a filepath and dataset name """
        self._filepath = filepath
        self._dataset_name = dataset_name

    def _load(self) -> Dataset:
        try:
            ds = Dataset.load_from_disk(self._filepath)
        except FileNotFoundError:
            ds = load_dataset(self._dataset_name)
        return ds

    def _save(self, data: Dataset) -> None:
        logger.info("Saving to local disk.")
        data.save_to_disk(self._filepath)

        logger.info("Saving to HuggingFace Hub")
        data.push_to_hub(self._dataset_name)

    def _describe(self) -> dict[str, Any]:
        api = HfApi()
        dataset_info = list(api.list_datasets(search=self._dataset_name))[0]

        return {
            "file_path": self._filepath,
            "dataset_name": self._dataset_name,
            "dataset_tags": dataset_info.tags,
            "dataset_author": dataset_info.author,
        }

    @staticmethod
    def list_datasets():
        api = HfApi()
        return list(api.list_datasets())
