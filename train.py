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
from huggingface_hub import Repository, get_full_repo_name
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
    GPT2Config,
)
from transformers.file_utils import get_full_repo_name
from transformers.testing_utils import CaptureLogger
from tokenizers import ByteLevelBPETokenizer



import os
import pickle

SEED=42

num_train_epochs = 1
per_device_train_batch_size = 64
per_device_eval_batch_size = 64

warmup_steps = 1000
learning_rate = 5e-3

block_size =512

logging_steps = 1 # 500
save_steps = 2500
eval_steps=2500

model_name = "gpt2_no"
output_dir = "gpt2_no"

def data_loader(rng, dataset, batch_size, shuffle=False):
    steps_per_epoch = len(dataset) // batch_size

    if shuffle:
        batch_idx = np.random.permutation(len(dataset))
    else:
        batch_idx = np.arange(len(dataset))

    batch_idx = batch_idx[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))

    for idx in batch_idx:
        batch = dataset[idx]
        batch = {k: np.array(v) for k, v in batch.items()}

        yield batch

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

class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray

    def replicate(self):
        return jax_utils.replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))

def write_train_metric(summary_writer, train_metrics, train_time, step):
    summary_writer.scalar("train_time", train_time, step)

    train_metrics = get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)

def write_eval_metric(summary_writer, eval_metrics, step):
    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval_{metric_name}", value, step)



def main():


    logging.basicConfig(filename="app.log", level =logging.INFO)
    logger = logging.getLogger(__name__)

    jax_devices = jax.device_count()


    print(jax.devices())

    print("-----setting up huggingface repo------")

    repo_name = get_full_repo_name(model_name)

    repo = Repository(output_dir, clone_from=repo_name)


    print("-------- Loading Dataset --------")

    dataset = load_dataset("oscar", "unshuffled_deduplicated_no")

    dataset["train"] = load_dataset("oscar", "unshuffled_deduplicated_no", split=f"train[:90%]")
    dataset["validation"] = load_dataset("oscar", "unshuffled_deduplicated_no", split=f"train[90%:]")

    column_names = dataset["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    print("-----Creating config----")

    if not os.path.exists("{output_dir}/config.json"):
        config = GPT2Config.from_pretrained("gpt2", resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, vocab_size=50257)
        config.save_pretrained(output_dir)
    else:
        print("---Loading pretrained config")
        config = AutoConfig.from_pretrained(output_dir)




    print("-------- Creating tokenizer --------")

    if not os.path.exists("{output_dir}/tokenizer.json"):

        tokenizer = ByteLevelBPETokenizer()

        def batch_iterator(batch_size=1000):
            for i in range(0, len(dataset), batch_size):
                yield dataset["train"][i: i + batch_size]["text"]

        # Customized training
        tokenizer.train_from_iterator(batch_iterator(), vocab_size=50257, min_frequency=2, special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ])

        # Save files to disk
        tokenizer.save(f"./{output_dir}/tokenizer.json")
    else:
        print("--Using cached tokenizer--")
        tokenizer = AutoTokenizer.from_pretrained({output_dir})

    print("-------- Tokenizing dataset --------")

    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    

    if not os.path.exists("cached_datasets/tokenized_dataset.pkl"):

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

        with open("cached_datasets/tokenized_dataset.pkl", "wb") as f:
            pickle.dump(lm_datasets, f)
    else:
        print("tokenized dataset on path, loading tokenized dataset")

        with open("cached_datasets/tokenized_dataset.pkl", "rb") as f:
            lm_datasets = pickle.load(f)


    print(f"-------- grouping dataset with block size {block_size}--------")

    if not os.path.exists("cached_datasets/grouped_dataset.pkl"):



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
            num_proc=8,
        )

        with open("grouped_dataset.pkl", "wb") as f:
            pickle.dump(lm_datasets, f)

    else:
        print("grouped dataset on path, loading grouped dataset")

        with open("cached_datasets/grouped_dataset.pkl", "rb") as f:
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

    print("--------setting up learning procedure--------")


    rng = jax.random.PRNGKey(SEED)
    rng, dropout_rng = jax.random.split(rng)

    num_epochs = int(num_train_epochs)
    train_batch_size = int(per_device_train_batch_size) * jax_devices
    eval_batch_size = int(per_device_eval_batch_size) * jax_devices
    steps_per_epoch = len(train_dataset) // train_batch_size
    total_train_steps = steps_per_epoch * num_epochs

    print("-----setting up learning rate scheduler-----")

    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        len(train_dataset),
        train_batch_size,
        num_train_epochs,
        warmup_steps,
        learning_rate,
    )

    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        flat_mask = {
            path: (path[-1] != "bias" and path[-2:] not in [("ln_1", "scale"), ("ln_2", "scale"), ("ln_f", "scale")])
            for path in flat_params
        }
        return traverse_util.unflatten_dict(flat_mask)

    print("-----setting up optimizer-----")

    optimizer = optax.adamw(
        learning_rate=linear_decay_lr_schedule_fn,
        b1=0.9,
        b2=0.98,
        eps= 1e-08,
        weight_decay=0.01,
        mask=decay_mask_fn,
    )


    print("---- Loading model-----")

    model = FlaxAutoModelForCausalLM.from_config(config, seed=SEED, dtype=getattr(jnp, "float32"))

    print("-----creating train state-----")

    state = TrainState.create(apply_fn=model.__call__, params=model.params, tx=optimizer, dropout_rng=dropout_rng)

    def loss_fn(logits, labels):
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        loss = optax.softmax_cross_entropy(shift_logits, onehot(shift_labels, shift_logits.shape[-1]))
        return loss.mean()

    def train_step(state, batch):
        dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

        def compute_loss(params):
            labels = batch.pop("labels")
            logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
            loss = loss_fn(logits, labels)
            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")

        new_state = state.apply_gradients(grads=grad, dropout_rng=new_dropout_rng)

        metrics = {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return new_state, metrics

    def eval_step(params, batch):
        labels = batch.pop("labels")
        logits = model(**batch, params=params, train=False)[0]
        loss = loss_fn(logits, labels)

        metrics = {"loss": loss}
        metrics = jax.lax.pmean(metrics, axis_name="batch")
        return metrics
      
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))
    p_eval_step = jax.pmap(eval_step, "batch")

    state = state.replicate()



    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {train_batch_size}")
    logger.info(f"  Total optimization steps = {total_train_steps}")


    train_time = 0
    train_metrics = []

    epochs = tqdm(range(num_epochs), desc="Epoch ...", position=0)

    for epoch in epochs:

        train_start = time.time() # Time of start of training

        rng, input_rng = jax.random.split(rng)

        train_loader = data_loader(input_rng, train_dataset, train_batch_size, shuffle=True)

        steps_per_epoch = len(train_dataset) // train_batch_size

        for step in tqdm(range(steps_per_epoch), desc="Training...", position=1, leave=False):
            batch = next(train_loader)
            batch = shard(batch) # Creates on-accelerator prefetch buffer (not neccesarry on TPUs)

            state, train_metric = p_train_step(state, batch)
            logging.info(f"Epoch {epoch}, Train step {step}")
            logging.info(train_metric)

            train_metrics.append(train_metric)

            cur_step = epoch * (len(train_dataset) // train_batch_size) + step

            if cur_step % logging_steps == 0 and cur_step > 0:
                train_metric = unreplicate(train_metric)
                train_time + time.time() - train_start

                if has_tensorboard and jax.process_index() == 0:
                    write_train_metric(summary_writer, train_metrics, train_time, cur_step)

                epochs.write( f"Step... ({cur_step} | Loss: {train_metric['loss'].mean()}, Learning Rate: {train_metric['learning_rate'].mean()})" )

                train_metrics = []

            if cur_step % 1000 == 0:
                # save checkpoint after each epoch and push checkpoint to the hub
                if jax.process_index() == 0:
                    params = jax.device_get(unreplicate(state.params))
                    model.save_pretrained(output_dir, params=params)
                    tokenizer.save_pretrained(output_dir)

                    commit_message = f"Commit after epoch {epoch}"

                    repo.push_to_hub(commit_message=commit_message, blocking=False)


if __name__ == '__main__':
    main()
