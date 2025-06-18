from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
import multiprocessing as mp
from itertools import chain


def get_datasets(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str,
    dataset_name: str,
    block_size: int = 1024,
):
    def preprocess_function(examples):
        tokenized = tokenizer([s + "\n\n" for s in examples["text"]])

        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    dataset_raw = load_dataset(dataset_path, dataset_name)
    tokenized_dataset = dataset_raw.map(
        preprocess_function,
        batched=True,
        num_proc=mp.cpu_count(),
        load_from_cache_file=True,
    ).remove_columns(["text"])
    dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=mp.cpu_count(),
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    return dataset
