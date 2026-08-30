from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class TokenStreamDataset(Dataset[tuple[Tensor, Tensor]]):
    """Non-overlapping ``block_size`` windows over a flat uint16 token file.

    The file is memory-mapped, so only the pages actually touched are read
    from disk and the OS page cache keeps hot pages in RAM. Indexing by window
    (not by token offset) keeps ``shuffle=True`` cheap for multi-billion-token
    streams. Only the path is stored: DataLoader workers receive a pickled copy
    of the dataset (Python >= 3.14 defaults to ``forkserver``), and pickling an
    open memmap would copy the whole file. Each process opens its own map lazily.
    """

    def __init__(self, path: Path, *, block_size: int) -> None:
        self.path = path
        self.block_size = block_size
        self._data: np.memmap | None = None

    @property
    def data(self) -> np.memmap:
        if self._data is None:
            self._data = np.memmap(self.path, dtype=np.uint16, mode="r")
        return self._data

    def __getstate__(self) -> dict:
        return {**self.__dict__, "_data": None}

    def __len__(self) -> int:
        # Each item needs block_size + 1 tokens (inputs plus shifted targets).
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        start = index * self.block_size
        # np.array copies out of the read-only memmap and widens to int64,
        # which is what nn.Embedding / cross_entropy expect.
        chunk = torch.from_numpy(np.array(self.data[start : start + self.block_size + 1], dtype=np.int64))
        return chunk[:-1], chunk[1:]


def get_owt_dataloaders(
    data_path: Path,
    *,
    block_size: int,
    batch_size: int,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    def make(split: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            TokenStreamDataset(data_path / f"{split}.bin", block_size=block_size),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=num_workers > 0,
        )

    return make("train", shuffle=True), make("val", shuffle=False)
