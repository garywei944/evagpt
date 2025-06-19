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
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["NCCL_IB_DISABLE"] = "0"
os.environ["NCCL_IB_HCA"] = "=mlx5_0,mlx5_1,mlx5_2,mlx5_3"
os.environ["NCCL_SOCKET_IFNAME"] = "=eth0,eth1,eth2,eth3"
os.environ["NCCL_ALGO"] = "Tree"
os.environ["NCCL_P2P_DISABLE"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only the first GPU

# jax
import jax
import jax.random as jrandom
from flax import nnx
import optax

# huggingface
from transformers import AutoTokenizer

from absl import logging
from tqdm import tqdm

from src.flax_gpt2 import *
from experiments.data.language_modeling.get_dataset import get_datasets
from experiments.data.language_modeling.jax_dataloader import get_dataloader


jax.config.update("jax_default_matmul_precision", "float32")
# jax.config.update("jax_disable_jit", True)

# hyperparameters
BATCH_SIZE_PER_DEVICE = 4
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 1
LOG_EVERY = 100
OUTPUT_DIR = "checkpoints/gpt2_single_gpu"


def loss_fn(model: GPT2, batch):
    logits, loss = model(**batch)

    return loss, logits


@nnx.jit
def train_step(model: GPT2, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(model, batch)
    metrics.update(loss=loss)
    optimizer.update(grads)


@nnx.jit
def eval_step(model: GPT2, batch):
    _, loss = model(**batch)
    metrics.update(loss=loss)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)

    n_devices = jax.device_count()

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
    model = GPT2(config, rngs=nnx.Rngs(0))

    optimizer = nnx.Optimizer(
        model,
        optax.adamw(
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
        ),
    )

    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))

    # dataset
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    datasets = get_datasets(
        tokenizer=tokenizer,
        dataset_path="wikitext",
        dataset_name="wikitext-103-raw-v1",
        block_size=config.block_size,
    )

    # training loop

    train_dataloader, train_steps_per_epoch = get_dataloader(
        datasets["train"],
        batch_size=BATCH_SIZE_PER_DEVICE * n_devices,
        epochs=NUM_EPOCHS,
        shuffle=True,
        drop_remainder=True,
        prefetch_size=4,
    )

    model.train()
    for step, batch in tqdm(
        enumerate(train_dataloader),
        total=NUM_EPOCHS * train_steps_per_epoch,
    ):
        train_step(model, optimizer, metrics, batch)
        for k, v in metrics.compute().items():
            logging.info(f"Step {step}: {k} = {v:.4f}")
        if step > 1000:
            break
