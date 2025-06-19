import os

# Set this to True to run the model on CPU only.
USE_CPU_ONLY = False

flags = os.environ.get("XLA_FLAGS", "")
if USE_CPU_ONLY:
    flags += " --xla_force_host_platform_device_count=8"  # Simulate 8 devices
    # Enforce CPU-only execution
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    # GPU flags
    flags += (
        "--xla_gpu_enable_triton_gemm=true "
        "--xla_gpu_enable_latency_hiding_scheduler=true "
        "--xla_gpu_enable_highest_priority_async_stream=true "
    )
os.environ["XLA_FLAGS"] = flags

import jax.random as jrandom
import jax.numpy as jnp
from flax import linen as nn
import tiktoken
from absl import logging

from src.jax_gpt2 import *

import jax

jax.config.update("jax_default_matmul_precision", "float32")
# jax.config.update("jax_disable_jit", True)

# Sanity check
# if __name__ == "__main__":
#     from flax.traverse_util import flatten_dict

#     config = GPT2Config()

#     key = jrandom.PRNGKey(0)
#     # x = jrandom.normal(key, (32, 1024))
#     x = jnp.arange(2 * config.block_size).reshape(2, -1)
#     model, params = GPT.from_pretrained("gpt2")

#     y = model.apply(params, x)
#     print(y.shape)  # Should be (2, 256, 65) for the given config
#     print(y)

#     gt_model = FlaxAutoModelForCausalLM.from_pretrained("gpt2")
#     gt_y = gt_model(x, params=gt_model.params).logits
#     print(gt_y.shape)  # Should be (2, 256, 50257) for the given config
#     print(gt_y)

#     diff = jnp.abs(y - gt_y)
#     print("max abs error:", float(jnp.max(diff)))
#     print("max rel error:", float(jnp.max(diff / (jnp.abs(gt_y) + 1e-8))))

# quick evaluation
if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    num_return_sequences = 5
    max_length = 30

    logging.info("Loading GPT-2 model for text generation...")
    config = GPT2Config()

    # model, params = GPT.from_pretrained("gpt2")
    model, params = GPT.from_config(config, seed=42)
    logging.info("Model loaded successfully.")

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("Hello, I'm a language model,")
    x = jnp.array(tokens).reshape(1, -1).repeat(num_return_sequences, axis=0)

    key = jrandom.PRNGKey(42)
    while x.shape[1] < max_length:
        logits = model.apply(params, x)  # (B, T, V)
        logits = logits[:, -1, :]  # (B, V)
        probs = nn.softmax(logits, axis=-1)  # (B, V)
        topk_probs, topk_indices = jax.lax.top_k(probs, k=50)  # (B, k), (B, k)
        key, subkey = jrandom.split(key)
        keys = jrandom.split(subkey, num=num_return_sequences)

        def sample_one(_key, _idx, _probs):
            return jrandom.choice(_key, _idx, p=_probs)

        next_token = jax.vmap(sample_one, in_axes=(0, 0, 0))(
            keys, topk_indices, topk_probs
        )  # (B,)
        x = jnp.concatenate([x, next_token[:, None]], axis=1)  # (B, T+1)

    for i in range(num_return_sequences):
        tokens = x[i, :].tolist()
        text = enc.decode(tokens)
        print(f"Generated sequence {i + 1}: {text}")
