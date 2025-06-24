import transformers
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.testing_utils import CaptureLogger
from datasets import load_dataset, DatasetDict

import multiprocessing as mp
from itertools import chain


tok_logger = transformers.utils.logging.get_logger(
    "transformers.tokenization_utils_base"
)


def get_datasets(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str,
    dataset_name: str,
    block_size: int = 1024,
) -> DatasetDict:
    def preprocess_function(examples):
        with CaptureLogger(tok_logger) as cl:
            return tokenizer([s + "\n\n" for s in examples["text"]])

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
    dataset = dataset_raw.map(
        preprocess_function,
        batched=True,
        num_proc=mp.cpu_count(),
        load_from_cache_file=True,
        remove_columns="text",
        desc=f"Tokenizing {dataset_name} dataset",
    ).map(
        group_texts,
        batched=True,
        num_proc=mp.cpu_count(),
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    return dataset
