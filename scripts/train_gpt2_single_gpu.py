import os

XLA_FLAGS = [
    # "--xla_gpu_enable_triton_gemm=true",
    # "--xla_gpu_enable_latency_hiding_scheduler=true",
    # "--xla_gpu_enable_highest_priority_async_stream=true",
    # "--xla_gpu_autotune_level=2",
    # "--xla_gpu_all_reduce_combine_threshold_bytes=16777216",
]
os.environ["XLA_FLAGS"] = " ".join(XLA_FLAGS)
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# os.environ["NCCL_IB_DISABLE"] = "0"
# os.environ["NCCL_IB_HCA"] = "=mlx5_0,mlx5_1,mlx5_2,mlx5_3"
# os.environ["NCCL_SOCKET_IFNAME"] = "=eth0,eth1,eth2,eth3"
# os.environ["NCCL_ALGO"] = "Tree"
# os.environ["NCCL_P2P_DISABLE"] = "0"

# jax
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import jit, value_and_grad
from flax import linen as nn
from flax.training import checkpoints
import optax
from jaxtyping import Array, Float

# torch
import torch
from torch.utils.data import DataLoader

# huggingface
from datasets import load_dataset
from transformers import AutoTokenizer

import numpy as np
from absl import logging
from tqdm import tqdm

from src.gpt2 import *
from experiments.data.language_modeling.get_dataset import get_datasets


jax.config.update("jax_default_matmul_precision", "float32")
jax.config.update("jax_disable_jit", True)

# hyperparameters
BATCH_SIZE = 2
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 10
LOG_EVERY = 100
OUTPUT_DIR = "checkpoints/gpt2_single_gpu"


def collate_fn(examples: list[dict[str, list[int]]]) -> dict[str, Float[Array, "B T"]]:
    return {k: jnp.array([e[k] for e in examples]) for k in examples[0].keys()}


@jit
def train_step(params, opt_state, x, y):
    def loss_fn(params):
        _, loss = model.apply(params, x, y)
        return loss

    grad_fn = value_and_grad(loss_fn)
    loss, grads = grad_fn(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    config = GPT2Config(
        block_size=1024,
        vocab_size=50257,  # GPT-2 vocab size
        n_layer=12,
        n_head=12,
        n_embd=768,
    )
    # model, params = GPT.from_pretrained("gpt2")
    model, params = GPT.from_config(config)

    # dataset
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    datasets = get_datasets(
        tokenizer=tokenizer,
        dataset_path="wikitext",
        dataset_name="wikitext-103-raw-v1",
        block_size=config.block_size,
    )
    data_loader = DataLoader(
        datasets["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        drop_last=True,
    )

    optimizer = optax.adamw(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        b1=0.9,
        b2=0.95,
        eps=1e-8,
    )
    opt_state = optimizer.init(params)

    # training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss = 0.0
        for step, batch in tqdm(enumerate(data_loader, start=1)):
            # for causal LM, inputs = targets
            x = batch["input_ids"]
            y = batch["input_ids"]

            params, opt_state, loss = train_step(params, opt_state, x, y)
            total_loss += float(loss)

            if step % LOG_EVERY == 0:
                logging.info(f"Epoch {epoch} | step {step} | loss = {loss:.4f}")

        avg_loss = total_loss / step
        logging.info(f"*** Epoch {epoch} completed. avg_loss = {avg_loss:.4f} ***")

        # save a checkpoint
        checkpoints.save_checkpoint(
            ckpt_dir=OUTPUT_DIR,
            target=params,
            step=epoch,
            overwrite=True,
        )
        logging.info(f"Saved checkpoint for epoch {epoch} in {OUTPUT_DIR}")
