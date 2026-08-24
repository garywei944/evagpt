"""Tokenize documents once, split whole documents by token budget, and write uint16 streams."""

import argparse
import fcntl
import gc
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import tiktoken

UINT16_DTYPE = np.dtype("<u2")
_ENCODINGS: dict[str, tiktoken.Encoding] = {}


def _get_encoding(name: str) -> tiktoken.Encoding:
    encoding = _ENCODINGS.get(name)
    if encoding is None:
        encoding = tiktoken.get_encoding(name)
        _ENCODINGS[name] = encoding
    return encoding


def _tokenize_batch(batch: dict[str, list[Any]], *, text_column: str, encoding_name: str) -> dict[str, Any]:
    encoding = _get_encoding(encoding_name)
    texts = batch[text_column]
    if not all(isinstance(text, str) for text in texts):
        raise TypeError(f"column {text_column!r} must contain only strings")

    rows = encoding.encode_batch(texts, num_threads=1, disallowed_special=())
    for row in rows:
        row.append(encoding.eot_token)
    return {"ids": rows, "length": [len(row) for row in rows]}


def _read_lengths(dataset: datasets.Dataset, batch_size: int) -> np.ndarray:
    lengths = np.empty(len(dataset), dtype=np.uint64)
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        lengths[start:stop] = np.asarray(dataset[start:stop]["length"], dtype=np.uint64)
    return lengths


def _choose_validation_documents(
    lengths: np.ndarray,
    *,
    target_tokens: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    if len(lengths) < 2:
        raise ValueError("at least two documents are required for a train/validation split")

    order = np.random.default_rng(seed).permutation(len(lengths))
    cumulative = np.cumsum(lengths[order], dtype=np.uint64)
    insertion = int(np.searchsorted(cumulative, target_tokens, side="left"))
    candidate_cuts = {
        max(1, min(len(lengths) - 1, insertion)),
        max(1, min(len(lengths) - 1, insertion + 1)),
    }
    cut = min(candidate_cuts, key=lambda index: abs(int(cumulative[index - 1]) - target_tokens))

    validation_mask = np.zeros(len(lengths), dtype=np.bool_)
    validation_mask[order[:cut]] = True
    return validation_mask, int(cumulative[cut - 1])


def _write_binary_splits(
    dataset: datasets.Dataset,
    *,
    validation_mask: np.ndarray,
    train_path: Path,
    validation_path: Path,
    train_tokens: int,
    validation_tokens: int,
    batch_size: int,
) -> None:
    train = np.memmap(train_path, dtype=UINT16_DTYPE, mode="w+", shape=(train_tokens,))
    validation = np.memmap(validation_path, dtype=UINT16_DTYPE, mode="w+", shape=(validation_tokens,))
    train_offset = 0
    validation_offset = 0

    total_batches = (len(dataset) + batch_size - 1) // batch_size
    for batch_index, start in enumerate(range(0, len(dataset), batch_size), start=1):
        stop = min(start + batch_size, len(dataset))
        rows = dataset[start:stop]["ids"]
        train_parts: list[np.ndarray] = []
        validation_parts: list[np.ndarray] = []

        for document_index, ids in enumerate(rows, start=start):
            part = np.asarray(ids, dtype=UINT16_DTYPE)
            if validation_mask[document_index]:
                validation_parts.append(part)
            else:
                train_parts.append(part)

        if train_parts:
            batch = np.concatenate(train_parts)
            train[train_offset : train_offset + len(batch)] = batch
            train_offset += len(batch)
        if validation_parts:
            batch = np.concatenate(validation_parts)
            validation[validation_offset : validation_offset + len(batch)] = batch
            validation_offset += len(batch)

        if batch_index == total_batches or batch_index % 100 == 0:
            print(f"writing binary splits: {batch_index}/{total_batches}")

    train.flush()
    validation.flush()
    del train
    del validation

    if train_offset != train_tokens or validation_offset != validation_tokens:
        raise RuntimeError(
            "written token counts do not match the planned split: "
            f"train={train_offset}/{train_tokens}, validation={validation_offset}/{validation_tokens}"
        )


def _prepare_into(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    cache_dir = output_dir / ".hf-cache"
    train_path = output_dir / "train.bin"
    validation_path = output_dir / "validation.bin"
    metadata_path = output_dir / "metadata.json"

    encoding = _get_encoding(args.encoding)
    if encoding.max_token_value > np.iinfo(UINT16_DTYPE).max:
        raise ValueError(f"encoding {args.encoding!r} does not fit in uint16")

    load_kwargs: dict[str, Any] = {
        "path": args.dataset,
        "name": args.dataset_config,
        "split": args.source_split,
        "num_proc": args.num_proc,
        "cache_dir": str(cache_dir),
    }
    if args.data_files is not None:
        load_kwargs["data_files"] = args.data_files
    raw = datasets.load_dataset(**load_kwargs)

    if args.max_documents is not None:
        raw = raw.select(range(min(args.max_documents, len(raw))))
    if args.text_column not in raw.column_names:
        raise KeyError(f"missing text column {args.text_column!r}; available columns: {raw.column_names}")

    features = datasets.Features(
        {
            "ids": datasets.List(datasets.Value("uint16")),
            "length": datasets.Value("uint64"),
        }
    )
    tokenized = raw.map(
        _tokenize_batch,
        batched=True,
        batch_size=args.map_batch_size,
        num_proc=args.num_proc,
        remove_columns=raw.column_names,
        features=features,
        fn_kwargs={"text_column": args.text_column, "encoding_name": args.encoding},
        desc="tokenizing documents",
    )

    lengths = _read_lengths(tokenized, args.write_batch_size)
    total_tokens = int(lengths.sum(dtype=np.uint64))
    if args.validation_tokens is not None:
        target_validation_tokens = args.validation_tokens
    else:
        target_validation_tokens = round(total_tokens * args.validation_fraction)
    target_validation_tokens = max(1, min(total_tokens - 1, target_validation_tokens))

    validation_mask, validation_tokens = _choose_validation_documents(
        lengths,
        target_tokens=target_validation_tokens,
        seed=args.seed,
    )
    train_tokens = total_tokens - validation_tokens

    _write_binary_splits(
        tokenized,
        validation_mask=validation_mask,
        train_path=train_path,
        validation_path=validation_path,
        train_tokens=train_tokens,
        validation_tokens=validation_tokens,
        batch_size=args.write_batch_size,
    )

    metadata = {
        "version": 1,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "source_split": args.source_split,
        "text_column": args.text_column,
        "encoding": args.encoding,
        "eot_token": encoding.eot_token,
        "dtype": UINT16_DTYPE.str,
        "seed": args.seed,
        "documents": len(tokenized),
        "train_documents": int((~validation_mask).sum()),
        "validation_documents": int(validation_mask.sum()),
        "tokens": total_tokens,
        "train_tokens": train_tokens,
        "validation_tokens": validation_tokens,
        "target_validation_tokens": target_validation_tokens,
        "validation_fraction": validation_tokens / total_tokens,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not args.keep_cache:
        del raw
        del tokenized
        gc.collect()
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
    return metadata


def prepare(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / f".{output_dir.name}.lock"

    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another process is preparing {output_dir}") from error

        if output_dir.exists():
            raise FileExistsError(f"output directory already exists; choose a fresh path: {output_dir}")

        staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
        try:
            metadata = _prepare_into(args, staging_dir)
            os.replace(staging_dir, output_dir)
        except BaseException:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

    print(json.dumps(metadata, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    default_workers = min(16, max(1, (os.cpu_count() or 1) // 2))
    parser = argparse.ArgumentParser(description="Prepare an EOT-delimited OpenWebText token stream.")
    parser.add_argument("--dataset", default="Skylion007/openwebtext")
    parser.add_argument("--dataset-config")
    parser.add_argument("--data-files", help="optional local file or glob passed to datasets.load_dataset")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--encoding", default="gpt2")
    parser.add_argument("--output-dir", type=Path, default=Path("data/openwebtext"))
    parser.add_argument("--validation-fraction", type=float, default=0.0005)
    parser.add_argument("--validation-tokens", type=int)
    parser.add_argument("--seed", type=int, default=2357)
    parser.add_argument("--num-proc", type=int, default=default_workers)
    parser.add_argument("--map-batch-size", type=int, default=1_000)
    parser.add_argument("--write-batch-size", type=int, default=1_000)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()

    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")
    if args.validation_tokens is not None and args.validation_tokens <= 0:
        parser.error("--validation-tokens must be positive")
    if args.num_proc <= 0 or args.map_batch_size <= 0 or args.write_batch_size <= 0:
        parser.error("worker and batch sizes must be positive")
    if args.max_documents is not None and args.max_documents <= 0:
        parser.error("--max-documents must be positive")
    return args


if __name__ == "__main__":
    prepare(parse_args())
