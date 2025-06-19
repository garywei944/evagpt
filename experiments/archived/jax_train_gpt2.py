import os

XLA_FLAGS = [
    "--xla_gpu_enable_triton_gemm=true",
    "--xla_gpu_enable_latency_hiding_scheduler=true",
    "--xla_gpu_enable_highest_priority_async_stream=true",
    "--xla_gpu_autotune_level=2",
    "--xla_gpu_all_reduce_combine_threshold_bytes=16777216",
]
os.environ["XLA_FLAGS"] = " ".join(XLA_FLAGS)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["NCCL_IB_DISABLE"] = "0"
os.environ["NCCL_IB_HCA"] = "=mlx5_0,mlx5_1,mlx5_2,mlx5_3"
os.environ["NCCL_SOCKET_IFNAME"] = "=eth0,eth1,eth2,eth3"
os.environ["NCCL_ALGO"] = "Tree"
os.environ["NCCL_P2P_DISABLE"] = "0"

# jax
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import jit, value_and_grad
from flax import jax_utils
from flax.training import train_state
from flax.training.common_utils import shard, shard_prng_key
import optax

# huggingface
from transformers import AutoTokenizer

import numpy as np
from absl import logging
from tqdm import tqdm

from src.jax_gpt2 import *
from experiments.data.language_modeling.get_dataset import get_datasets, data_loader


jax.config.update("jax_default_matmul_precision", "float32")
jax.config.update("jax_disable_jit", True)

# hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 10
LOG_EVERY = 100
OUTPUT_DIR = "checkpoints/gpt2_single_gpu"


class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray

    def replicate(self):
        return jax_utils.replicate(self).replace(
            dropout_rng=shard_prng_key(self.dropout_rng)
        )


def loss_fn(x, y):
    shift_x = x[..., :-1, :]
    shift_y = y[..., 1:]
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=shift_x, labels=shift_y
    )
    return loss.mean()


@jit
def train_step(state, batch):
    dropout_rng, new_rng = jrandom.split(state.dropout_rng)

    def compute_loss(params):
        labels = batch.pop("labels")
        logits = state.apply_fn(
            **batch, params=params, dropout_rng=dropout_rng, train=True
        )
        return loss_fn(logits, labels)

    grad_fn = value_and_grad(compute_loss)
    loss, grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")

    new_state = state.apply_gradients(grads=grads, dropout_rng=new_rng)

    return new_state, loss


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)

    rng = jrandom.PRNGKey(0)

    config = GPT2Config(
        block_size=1024,
        vocab_size=50257,  # GPT-2 vocab size
        n_layer=12,
        n_head=12,
        n_embd=768,
    )
    # model, params = GPT.from_pretrained("gpt2")
    rng, init_rng = jrandom.split(rng)
    model, params = GPT.from_config(config, rng=init_rng)

    # dataset
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    datasets = get_datasets(
        tokenizer=tokenizer,
        dataset_path="wikitext",
        dataset_name="wikitext-103-raw-v1",
        block_size=config.block_size,
    )

    optimizer = optax.adamw(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        b1=0.9,
        b2=0.95,
        eps=1e-8,
    )
    opt_state = optimizer.init(params)

    rng, dropout_rng = jrandom.split(rng)
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer, dropout_rng=dropout_rng
    )
    # training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        rng, input_rng = jrandom.split(rng)

        train_loader = data_loader(
            input_rng, datasets["train"], BATCH_SIZE, shuffle=True, drop_last=True
        )
        steps_per_epoch = len(datasets["train"]) // BATCH_SIZE

        for step in tqdm(
            range(steps_per_epoch),
            desc=f"Epoch {epoch}/{NUM_EPOCHS}",
            total=steps_per_epoch,
        ):
            batch = next(train_loader)
            batch = shard(batch)

            state, loss = train_step(state, batch)

            print(f"Step {step + 1}/{steps_per_epoch}, Loss: {loss:.4f}")
