import logging

import attrs
import equinox as eqx
import jax
import jax.random as jrandom

logger = logging.getLogger(__name__)


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


class GPT2(eqx.Module):
    config: GPTConfig
    wte: eqx.nn.Embedding
    wpe: eqx.nn.Embedding
    drop: eqx.nn.Dropout
    h: list[eqx.Module]
    ln_f: eqx.nn.LayerNorm

    def __init__(self, config: GPTConfig, *, key: jrandom.PRNGKey):
        super().__init__()

        keys = jrandom.split(key, 2 + config.n_layers)
        init = jax.nn.initializers.truncated_normal(0.02)

        self.config = config
        self.wte = eqx.nn.Embedding(weight=init(keys[0], (config.vocab_size, config.n_embd)))
        self.wpe = eqx.nn.Embedding(weight=init(keys[1], (config.block_size, config.n_embd)))
        self.drop = eqx.nn.Dropout(p=config.dropout)
        # self.h = [TransformerBlock(config=config, key=keys[i + 2]) for i in range(L)]
        self.ln_f = eqx.nn.LayerNorm(shape=config.n_embd, use_bias=config.bias)
