from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class TokenStreamDataset(Dataset[dict[str, Tensor]]):
    """Fixed-stride next-token windows over a uint16 token stream."""

    def __init__(self, path: str | Path, *, block_size: int):
        self.path = Path(path)
        self.block_size = block_size
        self._tokens: np.memmap | None = None

        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if self.path.stat().st_size % np.dtype("<u2").itemsize != 0:
            raise ValueError(f"token file has an invalid byte length: {self.path}")

        self.num_tokens = self.path.stat().st_size // np.dtype("<u2").itemsize
        self.num_samples = (self.num_tokens - 1) // block_size
        if self.num_samples == 0:
            raise ValueError(f"token file is too short for block_size={block_size}: {self.path}")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if index < 0:
            index += self.num_samples
        if not 0 <= index < self.num_samples:
            raise IndexError(index)

        start = index * self.block_size
        stop = start + self.block_size + 1
        window = np.asarray(self._memmap()[start:stop], dtype=np.int64)
        tokens = torch.from_numpy(window)
        return {"input_ids": tokens[:-1], "targets": tokens[1:]}

    def _memmap(self) -> np.memmap:
        if self._tokens is None:
            self._tokens = np.memmap(self.path, dtype="<u2", mode="r")
        return self._tokens

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_tokens"] = None
        return state
