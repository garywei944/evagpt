import itertools
import logging
import os
from pathlib import Path

import datasets
import tiktoken
from tap import Tap

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
NPROC = os.cpu_count() or 1


class Args(Tap):
    dataset_name: str = "Skylion007/openwebtext"
    dataset_config_name: str | None = None
    block_size: int = 1024
    batch_size: int = 8_000
    num_workers: int = NPROC // 2

    data_path: Path = PROJECT_ROOT / "data" / "processed"


def preprocess_dataset(
    dataset_name: str = "Skylion007/openwebtext",
    dataset_config_name: str | None = None,
    block_size: int = 1024,
    batch_size: int = 8_000,
    num_workers: int = NPROC // 2,
):
    logger.info("Loading dataset %s with config %s", dataset_name, dataset_config_name)
    tokenizer = tiktoken.get_encoding("gpt2")

    def tokenize(examples):
        batch_ids = tokenizer.encode_batch(examples["text"], disallowed_special=())
        for ids in batch_ids:
            ids.append(tokenizer.eot_token)
        return {"input_ids": batch_ids}

    def group_texts(examples):
        ids = list(itertools.chain.from_iterable(examples["input_ids"]))
        total_length = (len(ids) // block_size) * block_size
        chunks = [ids[i : i + block_size] for i in range(0, total_length, block_size)]
        return {"input_ids": chunks, "labels": chunks}

    raw_datasets = datasets.load_dataset(dataset_name, dataset_config_name, num_proc=num_workers)

    if "validation" not in raw_datasets:
        logger.info("No validation split found, creating from train split")
        raw_datasets = raw_datasets["train"].train_test_split(
            test_size=0.0005, seed=2357, shuffle=True
        )

    tokenized_datasets = raw_datasets.map(
        tokenize,
        batched=True,
        batch_size=batch_size,
        num_proc=num_workers,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing dataset",
    )
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=batch_size,
        num_proc=num_workers,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    return lm_datasets


if __name__ == "__main__":
    args = Args().parse_args()
    logger.info("Starting preprocessing with args: %s", args)

    ds = preprocess_dataset(
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        block_size=args.block_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    ds_path = args.data_path / f"owt_gpt2_bs{args.block_size}"
    ds_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(ds_path)
    logger.info("Saved processed dataset to %s", ds_path)
