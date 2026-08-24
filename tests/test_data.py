import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from evagpt.data import TokenStreamDataset


class TokenStreamDatasetTest(unittest.TestCase):
    def test_returns_shifted_fixed_stride_windows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.bin"
            np.arange(10, dtype="<u2").tofile(path)

            dataset = TokenStreamDataset(path, block_size=4)

            self.assertEqual(len(dataset), 2)
            torch.testing.assert_close(dataset[0]["input_ids"], torch.tensor([0, 1, 2, 3]))
            torch.testing.assert_close(dataset[0]["targets"], torch.tensor([1, 2, 3, 4]))
            torch.testing.assert_close(dataset[1]["input_ids"], torch.tensor([4, 5, 6, 7]))
            torch.testing.assert_close(dataset[1]["targets"], torch.tensor([5, 6, 7, 8]))

    def test_default_collation_produces_contiguous_batches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.bin"
            np.arange(10, dtype="<u2").tofile(path)
            loader = DataLoader(TokenStreamDataset(path, block_size=4), batch_size=2)

            batch = next(iter(loader))

            self.assertEqual(batch["input_ids"].shape, (2, 4))
            self.assertEqual(batch["targets"].shape, (2, 4))
            self.assertTrue(batch["input_ids"].is_contiguous())
            self.assertTrue(batch["targets"].is_contiguous())

    def test_memmap_reopens_in_worker_processes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.bin"
            np.arange(100, dtype="<u2").tofile(path)
            loader = DataLoader(TokenStreamDataset(path, block_size=8), batch_size=4, num_workers=2)

            batch = next(iter(loader))

            self.assertEqual(batch["input_ids"].shape, (4, 8))
            self.assertEqual(batch["targets"].shape, (4, 8))


if __name__ == "__main__":
    unittest.main()
