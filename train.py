import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import datasets
import numpy as np
from datasets import Dataset, load_dataset
from tqdm import tqdm

import jax
import jax.numpy as jnp
import optax
import transformers
from flax import jax_utils, traverse_util
from flax.jax_utils import unreplicate
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForCausalLM,
    HfArgumentParser,
    TrainingArguments,
    is_tensorboard_available,
    set_seed,
)
from transformers.file_utils import get_full_repo_name
from transformers.testing_utils import CaptureLogger

import os
import pickle

SEED=42

num_train_epochs = 20
per_device_train_batch_size = 64
per_device_eval_batch_size = 64

warmup_steps = 1000
learning_rate = 5e-3


def create_learning_rate_fn(
    train_ds_size, train_batch_size, num_train_epochs, num_warmup_steps, learning_rate):

    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn


def main():

    logger = logging.getLogger(__name__)

    jax_devices = jax.device_count()


    print(jax.devices())

    print("-------- Loading Dataset --------")

    dataset = load_dataset("oscar", "unshuffled_deduplicated_no")

    dataset["train"] = load_dataset("oscar", "unshuffled_deduplicated_no", split=f"train[:90%]")
    dataset["validation"] = load_dataset("oscar", "unshuffled_deduplicated_no", split=f"train[90%:]")

    column_names = dataset["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    print("Loading config")

    config = AutoConfig.from_pretrained("navjordj/gpt2_no")


    print("-------- Loading tokenizer --------")


    tokenizer = AutoTokenizer.from_pretrained("navjordj/gpt2_no")

    print("-------- Tokenizing dataset --------")

    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    

    if not os.path.exists("tokenized_dataset.pkl"):

        def tokenize_function(examples):
            with CaptureLogger(tok_logger) as cl:
                output = tokenizer(examples[text_column_name])
            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
                )
            return output

        lm_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=True,
        )

        with open("tokenized_dataset.pkl", "wb") as f:
            pickle.dump(lm_datasets, f)
    else:
        print("tokenized dataset on path, loading tokenized dataset")

        with open("tokenized_dataset.pkl", "rb") as f:
            lm_datasets = pickle.load(f)


    block_size = config.max_position_embeddings

    print("-------- grouping dataset --------")

    if not os.path.exists("grouped_dataset.pkl"):



        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result


        lm_datasets = lm_datasets.map(
            group_texts,
            batched=True,
            # num_proc= None,
        )

        with open("grouped_dataset.pkl", "wb") as f:
            pickle.dump(lm_datasets, f)

    else:
        print("grouped dataset on path, loading grouped dataset")

        with open("grouped_dataset.pkl", "rb") as f:
            lm_datasets = pickle.load(f)

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    has_tensorboard = is_tensorboard_available()
    if has_tensorboard and jax.process_index() == 0:
        try:
            from flax.metrics.tensorboard import SummaryWriter
            print("using SummaryWriter for logging")
            summary_writer = SummaryWriter(log_dir=Path("summary/"))
        except ImportError as ie:
            has_tensorboard = False
            logger.warning(
                f"Unable to display metrics through TensorBoard because some package are not installed: {ie}"
            )
    else:
        logger.warning(
            "Unable to display metrics through TensorBoard because the package is not installed: "
            "Please run pip install tensorboard to enable."
        )

    rng = jax.random.PRNGKey(SEED)
    rng, dropout_rng = jax.random.split(rng)

    num_epochs = int(num_train_epochs)
    train_batch_size = int(per_device_train_batch_size) * jax_devices
    eval_batch_size = int(per_device_eval_batch_size) * jax_devices
    steps_per_epoch = len(train_dataset) // train_batch_size
    total_train_steps = steps_per_epoch * num_epochs


    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        len(train_dataset),
        train_batch_size,
        num_train_epochs,
        warmup_steps,
        learning_rate,
    )





if __name__ == '__main__':
    main()