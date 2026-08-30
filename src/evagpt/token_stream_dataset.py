from pathlib import Path

import numpy as np
import torch
from jaxtyping import Int64, UInt16
from torch import Tensor
from torch.utils import data as torch_data


class TokenStreamDataset(torch_data.Dataset[tuple[Int64[Tensor, " T"], Int64[Tensor, " T"]]]):
    def __init__(self, path: Path, block_size: int):
        super().__init__()

        self.path = path
        self.block_size = block_size
        self._data = None

    @property
    def data(self) -> UInt16[np.memmap, " total_tokens"]:
        if self._data is None:
            self._data = np.memmap(self.path, dtype=np.uint16, mode="r")
        return self._data

    def __getstate__(self) -> dict:
        return {**self.__dict__, "_data": None}

    def __len__(self) -> int:
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, index: int) -> tuple[Int64[Tensor, " T"], Int64[Tensor, " T"]]:
        start = index * self.block_size

        chunk = torch.from_numpy(np.array(self.data[start : start + self.block_size + 1], dtype=np.int64))

        return chunk[:-1], chunk[1:]


def get_dataloader(
    data_file: Path | str, *, block_size: int, batch_size: int, shuffle: bool = True, num_workers: int = 4
) -> torch_data.DataLoader:
    dataset = TokenStreamDataset(path=Path(data_file), block_size=block_size)

    return torch_data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
