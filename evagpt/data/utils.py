import math
from typing import Any, Generator

import datasets
import flax.jax_utils
import flax.training.common_utils
import jax
import tensorflow as tf

SHUFFLE_BUFFER_SIZE = 10_000


def get_dataloader(
    dataset: datasets.Dataset,
    batch_size: int,
    epochs: int = 1,
    shuffle: bool = True,
    drop_last: bool = True,
    prefetch_size: int = 4,
) -> tuple[Generator[Any, None, None], int]:
    n = dataset.num_rows

    if jax.process_count() > 1:
        dataset = dataset.shard(num_shards=jax.process_count(), index=jax.process_index())

    ds = dataset.to_tf_dataset(
        batch_size=batch_size,
        columns=["input_ids", "labels"],
        shuffle=shuffle,
        drop_remainder=drop_last,
        prefetch=False,
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)
    ds = ds.cache().repeat(epochs).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()

    steps_per_epoch = n // batch_size if drop_last else math.ceil(n / batch_size)

    sharded = map(flax.training.common_utils.shard, ds)

    return flax.jax_utils.prefetch_to_device(sharded, prefetch_size), steps_per_epoch
