from flax.jax_utils import prefetch_to_device
from flax.training.common_utils import shard
from jaxtyping import Int, Array

import tensorflow as tf

from datasets import Dataset

import math
from typing import Generator

SHUFFLE_BUFFER_SIZE = 10_000

__all__ = ["get_dataloader"]


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    epochs: int = 1,
    shuffle: bool = True,
    drop_remainder: bool = True,
    prefetch_size: int = 4,
) -> tuple[
    Generator[tuple[dict[str, Int[Array, "B T"]], Int[Array, "B T"]], None, None], int
]:
    ds = dataset.to_tf_dataset(
        batch_size=batch_size,
        columns=["input_ids", "attention_mask", "labels"],
        drop_remainder=drop_remainder,
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)
    ds = ds.cache().repeat(epochs).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()

    steps_per_epoch = (
        dataset.num_rows // batch_size
        if drop_remainder
        else math.ceil(dataset.num_rows / batch_size)
    )

    return prefetch_to_device(ds, prefetch_size), steps_per_epoch
