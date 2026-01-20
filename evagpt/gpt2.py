import logging
import math

import attrs
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Int, PRNGKeyArray, jaxtyped

__all__ = ["GPTConfig", "GPT2"]

logger = logging.getLogger(__name__)

std_init = jax.nn.initializers.truncated_normal(0.02)


@attrs.define
class GPTConfig:
    block_size: int = 1024
    # GPT-2 vocab size of 50257, rounded up to nearest multiple of 64 for efficiency
    vocab_size: int = 50304
    n_layers: int = 12
    n_heads: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True
    dtype: jnp.dtype = jnp.bfloat16


def init_linear(
    in_features: int,
    out_features: int,
    use_bias: bool = True,
    dtype: jnp.dtype = jnp.float32,
    stddev: float = 0.02,
    *,
    key: PRNGKeyArray,
):
    linear = eqx.nn.Linear(in_features, out_features, use_bias=use_bias, dtype=dtype, key=key)
    linear = eqx.tree_at(
        lambda m: m.weight,
        linear,
        std_init(key, (out_features, in_features), dtype=dtype) * stddev / 0.02,
    )
    if use_bias:
        linear = eqx.tree_at(lambda m: m.bias, linear, jnp.zeros((out_features,), dtype=dtype))
    return linear


class MLP(eqx.Module):
    config: GPTConfig
    c_fc: eqx.nn.Linear
    c_proj: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(self, config: GPTConfig, *, key: PRNGKeyArray):
        super().__init__()

        self.config = config

        key_fc, key_proj = jrandom.split(key, 2)
        self.c_fc = init_linear(
            config.n_embd, 4 * config.n_embd, use_bias=config.bias, dtype=config.dtype, key=key_fc
        )
        self.c_proj = init_linear(
            4 * config.n_embd,
            config.n_embd,
            use_bias=config.bias,
            dtype=config.dtype,
            stddev=0.02 / math.sqrt(2 * config.n_layers),
            key=key_proj,
        )
        self.dropout = eqx.nn.Dropout(config.dropout)

    @jaxtyped(typechecker=typechecker)
    def __call__(self, x: Float[Array, "B T C"], *, key: PRNGKeyArray) -> Float[Array, "B T C"]:
        x = jax.vmap(jax.vmap(self.c_fc))(x)
        x = jax.nn.gelu(x)
        x = jax.vmap(jax.vmap(self.c_proj))(x)
        x = self.dropout(x, key=key)
        return x


class CausalSelfAttention(eqx.Module):
    config: GPTConfig

    c_attn: eqx.nn.Linear
    c_proj: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(self, config: GPTConfig, *, key: PRNGKeyArray):
        super().__init__()

        self.config = config

        key_attn, key_proj = jrandom.split(key, 2)
        self.c_attn = init_linear(
            config.n_embd, 3 * config.n_embd, use_bias=config.bias, dtype=config.dtype, key=key_attn
        )
        self.c_proj = init_linear(
            config.n_embd,
            config.n_embd,
            use_bias=config.bias,
            stddev=0.02 / math.sqrt(2 * config.n_layers),
            dtype=config.dtype,
            key=key_proj,
        )
        self.dropout = eqx.nn.Dropout(config.dropout)

    @jaxtyped(typechecker=typechecker)
    def __call__(self, x: Float[Array, "B T C"], *, key: PRNGKeyArray) -> Float[Array, "B T C"]:
        B, T, C = x.shape
        nh, hs = self.config.n_heads, C // self.config.n_heads

        q, k, v = jnp.split(jax.vmap(jax.vmap(self.c_attn))(x), 3, axis=-1)
        q = q.reshape(B, T, nh, hs)
        k = k.reshape(B, T, nh, hs)
        v = v.reshape(B, T, nh, hs)

        # out: (B, T, nh, hs)
        out = jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation="cudnn")
        out = out.reshape(B, T, C)
        out = self.dropout(out, key=key)

        return out


class Block(eqx.Module):
    ln1: eqx.nn.LayerNorm
    attn: CausalSelfAttention
    ln2: eqx.nn.LayerNorm
    mlp: MLP

    def __init__(self, config: GPTConfig, *, key: PRNGKeyArray):
        super().__init__()
        key_attn, key_mlp = jrandom.split(key, 2)
        self.ln1 = eqx.nn.LayerNorm(shape=config.n_embd, use_bias=config.bias, dtype=config.dtype)
        self.attn = CausalSelfAttention(config=config, key=key_attn)
        self.ln2 = eqx.nn.LayerNorm(shape=config.n_embd, use_bias=config.bias, dtype=config.dtype)
        self.mlp = MLP(config=config, key=key_mlp)

    @jaxtyped(typechecker=typechecker)
    def __call__(self, x: Float[Array, "B T C"], *, key: PRNGKeyArray) -> Float[Array, "B T C"]:
        key_attn, key_mlp = jrandom.split(key, 2)
        x = x + self.attn(jax.vmap(jax.vmap(self.ln1))(x), key=key_attn)
        x = x + self.mlp(jax.vmap(jax.vmap(self.ln2))(x), key=key_mlp)
        return x


class GPT2(eqx.Module):
    config: GPTConfig

    wte: eqx.nn.Embedding
    wpe: eqx.nn.Embedding
    dropout: eqx.nn.Dropout
    h: list[Block]
    ln_f: eqx.nn.LayerNorm

    def __init__(self, config: GPTConfig, *, key: PRNGKeyArray):
        super().__init__()

        self.config = config

        keys = jrandom.split(key, 2 + config.n_layers)
        self.wte = eqx.nn.Embedding(
            weight=init_linear(
                config.n_embd, config.vocab_size, use_bias=False, dtype=config.dtype, key=keys[0]
            ).weight
        )
        self.wpe = eqx.nn.Embedding(
            weight=init_linear(
                config.n_embd, config.block_size, use_bias=False, dtype=config.dtype, key=keys[1]
            ).weight
        )
        self.dropout = eqx.nn.Dropout(config.dropout)
        self.h = [Block(config=config, key=keys[i + 2]) for i in range(config.n_layers)]
        self.ln_f = eqx.nn.LayerNorm(shape=config.n_embd, use_bias=config.bias, dtype=config.dtype)

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self,
        input_ids: Int[Array, "B T"],
        labels: Int[Array, "B T"] | None = None,
        *,
        key: PRNGKeyArray,
    ) -> tuple[Float[Array, "B T V"], Float[Array, ""] | None]:
        _, T = input_ids.shape
        keys = jrandom.split(key, self.config.n_layers + 1)

        tok_emb = jax.vmap(jax.vmap(self.wte))(input_ids)
        pos_emb = jax.vmap(self.wpe)(jnp.arange(T))
        x = self.dropout(tok_emb + pos_emb, key=keys[0])

        for i, block in enumerate(self.h):
            x = block(x, key=keys[i + 1])
        x = jax.vmap(jax.vmap(self.ln_f))(x)

        # logits: (B, T, C)
        logits = jnp.einsum("btc,vc->btv", x, self.wte.weight)

        # Compute loss
        if labels is None:
            return logits, None

        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        loss = optax.softmax_cross_entropy_with_integer_labels(shift_logits, shift_labels)
        return logits, loss.mean()


if __name__ == "__main__":
    key = jrandom.key(0)
    config = GPTConfig()

    key, key_model, key_data = jrandom.split(key, 3)

    model = GPT2(config=config, key=key_model)
    x = jrandom.randint(key_data, (2, 1024), 0, config.vocab_size)
    y = x

    logits, loss = model(x, labels=y, key=key)
    print("logits shape:", logits.shape)
    print("loss:", loss)

    print(logits[:, :10, :10])

    # # mlp = MLP(config=config, key=key_model)
    # # attn = CausalSelfAttention(config=config, key=key_model)
    # block = Block(config=config, key=key_model)
    # x = jrandom.normal(key_data, (2, 1024, config.n_embd), dtype=config.dtype)
    # key, key_mlp = jrandom.split(key, 2)
    # y = block(x, key=key_mlp)
    # print("Block output shape:", y.shape)
    # print(y)
