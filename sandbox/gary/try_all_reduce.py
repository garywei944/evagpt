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

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import lax, pmap
from flax import linen as nn
import tiktoken
from absl import logging

from src.gpt2 import *


jax.config.update("jax_default_matmul_precision", "float32")
# jax.config.update("jax_disable_jit", True)


def allreduce_once(x):
    # sums x across all devices on axis “i”
    return lax.psum(x, axis_name="i")


if __name__ == "__main__":
    # prepare one array per local device
    ndev = jax.local_device_count()
    x = jnp.arange(16 * ndev, dtype=jnp.float32).reshape((ndev, 16))
    # run a single all-reduce
    y = pmap(allreduce_once, axis_name="i")(x)
    print("per-device input:\n", x)
    print("after lax.psum across", ndev, "devices:\n", y)
