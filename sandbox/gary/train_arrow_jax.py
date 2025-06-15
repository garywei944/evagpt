#!/usr/bin/env python3
"""
train_arrow_jax.py

A minimal demo of HuggingFace Datasets + Arrow + JAX pipeline.
"""

import argparse
from datasets import load_dataset
from transformers import GPT2TokenizerFast, default_data_collator
import jax
import jax.numpy as jnp
import flax.jax_utils
import numpy as np


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="openwebtext")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--batch_per_device", type=int, default=8)
    p.add_argument("--num_proc", type=int, default=8)
    p.add_argument("--shuffle_buffer", type=int, default=100_000)
    return p.parse_args()


def tokenize_and_pack(examples, tokenizer, seq_len):
    tok = tokenizer(
        examples["text"],
        truncation=True,
        max_length=seq_len,
        padding="max_length",
        return_attention_mask=False,
    )
    return {"input_ids": tok["input_ids"]}


def make_data_pipeline(args, tokenizer):
    # 1) load raw dataset
    ds = load_dataset(args.dataset, split=args.split)
    # 2) shuffle, tokenize in parallel & cache Arrow
    ds = ds.shuffle(args.shuffle_buffer)
    ds = ds.map(
        lambda ex: tokenize_and_pack(ex, tokenizer, args.max_length),
        batched=True,
        batch_size=1_000,
        num_proc=args.num_proc,
        remove_columns=["text"],
        load_from_cache_file=True,
    )
    # 3) memory-map Arrow → NumPy arrays
    ds = ds.with_format("numpy", columns=["input_ids"])

    # 4) generator yielding device-sharded batches
    ndev = jax.local_device_count()
    global_bs = args.batch_per_device * ndev

    def gen():
        for batch in ds.batch(global_bs, drop_remainder=True):
            arr = batch["input_ids"].reshape(
                (ndev, args.batch_per_device, args.max_length)
            )
            yield {"input_ids": jnp.array(arr)}

    # 5) prefetch onto devices
    return flax.jax_utils.prefetch_to_device(gen(), 2)


@jax.pmap
def train_step(batch):
    # dummy train: return sum of tokens mod 1e6
    x = batch["input_ids"]
    return jnp.sum(x) % 1_000_000


def main():
    args = get_args()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    data_iter = make_data_pipeline(args, tokenizer)

    print(
        f"Running dummy train over {args.split} split on {jax.local_device_count()} devices"
    )
    for i, batch in enumerate(data_iter):
        loss = train_step(batch)
        print(f"Step {i:04d}: loss = {loss[0]}")
        if i >= 5:
            break


if __name__ == "__main__":
    main()
