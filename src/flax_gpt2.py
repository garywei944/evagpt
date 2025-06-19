from flax import nnx
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int, Scalar, jaxtyped
from beartype import beartype as typechecker


from dataclasses import dataclass

__all__ = ["GPT2Config", "MLP", "CausalSelfAttention", "Block", "GPT2"]


@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50257  # GPT-2 vocab size
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class MLP(nnx.Module):
    def __init__(self, config: GPT2Config, *, rngs: nnx.Rngs):
        self.c_fc = nnx.Linear(config.n_embd, config.n_embd * 4, rngs=rngs)
        self.c_proj = nnx.Linear(config.n_embd * 4, config.n_embd, rngs=rngs)

    @jaxtyped(typechecker=typechecker)
    def __call__(self, x: Float[Array, "*B T C"]) -> Float[Array, "*B T C"]:
        x = self.c_fc(x)
        x = nnx.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nnx.Module):
    def __init__(self, config: GPT2Config, *, rngs: nnx.Rngs):
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # q, k, v for all heads
        self.c_attn = nnx.Linear(config.n_embd, config.n_embd * 3, rngs=rngs)
        # output projection
        self.c_proj = nnx.Linear(config.n_embd, config.n_embd, rngs=rngs)

    @jaxtyped(typechecker=typechecker)
    def __call__(self, x: Float[Array, "*B T C"]) -> Float[Array, "*B T C"]:
        *B, T, C = x.shape
        hs = C // self.n_head

        qkv = self.c_attn(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(*B, T, self.n_head, hs)  # (*B, T, nh, hs)
        k = k.reshape(*B, T, self.n_head, hs)  # (*B, T, nh, hs)
        v = v.reshape(*B, T, self.n_head, hs)  # (*B, T, nh, hs)

        # att = jnp.einsum("bihc,bjhc->bhij", q, k, precision="high")
        att = q.swapaxes(-2, -3) @ k.swapaxes(-2, -3).swapaxes(-1, -2)  # (*B, nh, T, T)
        att /= jnp.sqrt(hs)
        mask = ~jnp.tril(jnp.ones((T, T), dtype=jnp.bool))
        att = jnp.where(mask, -jnp.inf, att)
        att = nnx.softmax(att, axis=-1)
        # y = jnp.einsum("bhij,bjhc->bihc", att, v, precision="high").reshape(*B, T, C)
        y = att @ v.swapaxes(-2, -3)  # (*B, nh, T, hs)
        y = y.swapaxes(-2, -3).reshape(*B, T, C)  # (*B, T, C)
        # output projection
        y = self.c_proj(y)

        return y


class Block(nnx.Module):
    def __init__(self, config: GPT2Config, *, rngs: nnx.Rngs):
        self.ln_1 = nnx.LayerNorm(config.n_embd, rngs=rngs)
        self.attn = CausalSelfAttention(config, rngs=rngs)
        self.ln_2 = nnx.LayerNorm(config.n_embd, rngs=rngs)
        self.mlp = MLP(config, rngs=rngs)

    @jaxtyped(typechecker=typechecker)
    def __call__(self, x: Float[Array, "*B T C"]) -> Float[Array, "*B T C"]:
        x += self.attn(self.ln_1(x))
        x += self.mlp(self.ln_2(x))
        return x


class GPT2(nnx.Module):
    def __init__(self, config: GPT2Config, *, rngs: nnx.Rngs):
        self.config = config
        self.wte = nnx.Embed(config.vocab_size, config.n_embd, rngs=rngs)
        self.wpe = nnx.Embed(config.block_size, config.n_embd, rngs=rngs)

        self.h = [Block(config, rngs=rngs) for _ in range(config.n_layer)]
        self.ln_f = nnx.LayerNorm(config.n_embd, rngs=rngs)

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self,
        input_ids: Int[Array, "*B T"],
        attention_mask: Int[Array, "*B T"] | None = None,
        labels: Int[Array, "*B T"] | None = None,
    ) -> tuple[Float[Array, "*B T C"], Scalar | None]:
        *_, T = input_ids.shape
        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(jnp.arange(T))
        x = tok_emb + pos_emb

        for block in self.h:
            x = block(x)
        x = self.ln_f(x)

        logits = x @ self.wte.embedding.T  # (B, T, V)
        loss = None

        if labels is not None:
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=shift_logits, labels=shift_labels
            )
            loss = loss.mean()
        return logits, loss


if __name__ == "__main__":
    from jax import random as jrandom
    from flax import nnx

    rng = jrandom.PRNGKey(0)
    config = GPT2Config(
        block_size=1024,
        vocab_size=50257,  # GPT-2 vocab size
        n_layer=12,
        n_head=12,
        n_embd=768,
    )
    model = GPT2(config, rngs=nnx.Rngs(0))
    # print(model)

    x = jnp.arange(2 * 1024, dtype=jnp.int32).reshape(2, 1024)
    y = x
    logits, loss = model(x, labels=y)
    print(logits[:, :10, :10])
    print("Loss:", loss)
