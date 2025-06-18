import jax.random as jrandom

from transformers import PreTrainedTokenizerBase
from datasets import load_dataset, DatasetDict, Dataset

import numpy as np
import math

import multiprocessing as mp
from itertools import chain


def get_datasets(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str,
    dataset_name: str,
    block_size: int = 1024,
) -> DatasetDict:
    def preprocess_function(examples):
        return tokenizer([s + "\n\n" for s in examples["text"]])

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    dataset_raw = load_dataset(dataset_path, dataset_name)
    dataset = dataset_raw.map(
        preprocess_function,
        batched=True,
        num_proc=mp.cpu_count(),
        load_from_cache_file=True,
        remove_columns="text",
        desc=f"Tokenizing {dataset_name} dataset",
    ).map(
        group_texts,
        batched=True,
        num_proc=mp.cpu_count(),
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    return dataset


def data_loader(
    rng: jrandom.PRNGKey,
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    drop_last=True,
):
    """
    Returns batches of size `batch_size` from `dataset`. If `drop_last` is set to `False`, the final batch may be incomplete,
    and range in size from 1 to `batch_size`. Shuffle batches if `shuffle` is `True`.
    """
    if shuffle:
        batch_idx = jrandom.permutation(rng, len(dataset))
        batch_idx = np.asarray(batch_idx)
    else:
        batch_idx = np.arange(len(dataset))

    if drop_last:
        steps_per_epoch = len(dataset) // batch_size
        batch_idx = batch_idx[: steps_per_epoch * batch_size]  # Skip incomplete batch.
        batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))
    else:
        steps_per_epoch = math.ceil(len(dataset) / batch_size)
        batch_idx = np.array_split(batch_idx, steps_per_epoch)

    for idx in batch_idx:
        batch = dataset[idx]
        batch = {k: np.array(v) for k, v in batch.items()}

        yield batch
