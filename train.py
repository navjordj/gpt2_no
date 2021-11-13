from datasets import load_dataset
import transformers

from transformers import AutoTokenizer, AutoConfig
from transformers.testing_utils import CaptureLogger

import os
import pickle

def main():

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
            tokenized_dataset = pickle.load(f)


    block_size = config.max_position_embeddings

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