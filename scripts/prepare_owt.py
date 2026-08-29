import logging
import os
from pathlib import Path
from typing import cast

import datasets
import numpy as np
import tap
import tiktoken
import tqdm

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
NPROC = os.cpu_count() or 1


class Args(tap.Tap):
    dataset_name: str = "Skylion007/openwebtext"
    dataset_config_name: str = ""
    num_workers: int = NPROC
    total_batches: int = 1024

    data_path: Path = PROJECT_ROOT / "data/processed/openwebtext"


def preprocess_dataset(
    dataset_name: str, dataset_config_name: str | None, total_batches: int, num_workers: int, data_path: Path
):
    logger.info("Loading dataset %s with config %s", dataset_name, dataset_config_name)
    tokenizer = tiktoken.encoding_for_model("gpt2")

    dataset = datasets.load_dataset(dataset_name, dataset_config_name, num_proc=num_workers)

    split_dataset = cast(
        datasets.DatasetDict,
        dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True),
    )

    tokenized = split_dataset.map(
        _tokenize,
        remove_columns=["text"],
        desc="tokenizing the splits",
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=num_workers,
    )

    data_path.mkdir(parents=True, exist_ok=True)

    for split, ds in tokenized.items():
        arr_len = int(np.sum(ds.with_format("numpy")["len"], dtype=np.uint64))

        data_file = data_path / f"{split}.bin"
        data = np.memmap(data_file, dtype=np.uint16, mode="w+", shape=(arr_len,))

        idx = 0
        for batch_idx in tqdm.tqdm(range(total_batches), desc=f"writing {data_file}"):
            batch = ds.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format("numpy")
            arr_batch = np.concatenate(batch["input_ids"])
            data[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        data.flush()


def _tokenize(example: dict[str, str], *, tokenizer: tiktoken.Encoding) -> dict[str, list[int] | int]:
    input_ids = [tokenizer.eot_token] + tokenizer.encode_ordinary(example["text"])
    return {"input_ids": input_ids, "len": len(input_ids)}


def main():
    logging.basicConfig(level=logging.INFO)

    args = Args().parse_args()
    logger.info("Starting preprocessing with args %s", args)

    preprocess_dataset(
        args.dataset_name,
        args.dataset_config_name,
        total_batches=args.total_batches,
        num_workers=args.num_workers,
        data_path=args.data_path,
    )


if __name__ == "__main__":
    main()
