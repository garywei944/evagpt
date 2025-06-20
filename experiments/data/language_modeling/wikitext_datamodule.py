import lightning as L
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataclasses import dataclass

from experiments.data.language_modeling.get_dataset import get_datasets


def collate_fn(examples: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
    return {k: torch.tensor([e[k] for e in examples]) for k in examples[0].keys()}


@dataclass
class WikiTextDataModule(L.LightningDataModule):
    tokenizer_name: str = "openai-community/gpt2"
    dataset_path: str = "wikitext"
    dataset_name: str = "wikitext-103-v1"
    train_batch_size_per_device: int = 8
    eval_batch_size_per_device: int = 8
    block_size: int = 1024

    def __post_init__(self):
        super().__init__()

        self.save_hyperparameters()

    def prepare_data(self):
        # ! Only rank 0 should download the data,
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        datasets = get_datasets(
            tokenizer, self.dataset_path, self.dataset_name, self.block_size
        ).remove_columns("attention_mask")

    def setup(self, stage: str):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.datasets = get_datasets(
            self.tokenizer, self.dataset_path, self.dataset_name, self.block_size
        )

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.train_batch_size_per_device,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.eval_batch_size_per_device,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.eval_batch_size_per_device,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )
