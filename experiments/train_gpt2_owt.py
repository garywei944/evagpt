from evagpt.utils import logging_utils

logging_utils.setup_root_logger()

import logging

logging.getLogger().setLevel(logging.INFO)


import functools

import datasets
import jax
import jax.numpy as jnp
import jax.random as jrandom
import tensorflow as tf

import evagpt.data.utils
import evagpt.gpt2

logger = logging.getLogger(__name__)


tf.config.set_visible_devices([], "GPU")

MODEL_NAME = "gpt2"
SEED = 42


@functools.partial(jax.pmap, axis_name="batch")
def train_step(model, batch, rng):
    logits, loss = model(**batch, key=rng)
    return loss, logits


def main():
    rng = jrandom.PRNGKey(SEED)
    config = evagpt.gpt2.GPT2Config.from_pretrained(MODEL_NAME)

    rng, init_rng, data_rng = jrandom.split(rng, 3)
    model = evagpt.gpt2.GPT2(config=config, key=init_rng)

    ds = datasets.load_from_disk("data/processed/owt_gpt2_bs1024")
    train_loader, steps_per_epoch = evagpt.data.utils.get_dataloader(
        ds["train"], batch_size=64, epochs=1, shuffle=True, drop_last=True
    )

    for batch in train_loader:
        logger.info(batch["input_ids"].shape)

        rng, *keys = jrandom.split(rng, jax.local_device_count() + 1)
        loss, logits = train_step(model, batch, jnp.stack(keys))

        # logger.info(f"Logits shape: {logits.shape}, Loss: {loss}")
        logger.info("logits.shape: %s", logits.shape)
        logger.info("logits: %s", logits)
        logger.info("loss: %s", loss)

        break


if __name__ == "__main__":
    main()
