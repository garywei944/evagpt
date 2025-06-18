import jax.random as jrandom
import jax.numpy as jnp
from flax import linen as nn
import tiktoken

from transformers import FlaxAutoModelForCausalLM

from src.gpt2 import *

import jax

jax.config.update("jax_default_matmul_precision", "float32")
# jax.config.update("jax_disable_jit", True)

if __name__ == "__main__":
    num_return_sequences = 5
    max_length = 30

    config = GPT2Config()

    model, params = GPT.from_pretrained("gpt2")

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
