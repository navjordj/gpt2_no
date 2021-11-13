from datasets import load_dataset
import transformers

from transformers import AutoTokenizer
from transformers.testing_utils import CaptureLogger

from multiprocessing import Process, freeze_support

def main():


    dataset = load_dataset("oscar", "unshuffled_deduplicated_no")
    print("1")
    dataset["train"] = load_dataset("oscar", "unshuffled_deduplicated_no", split=f"train[:90%]")
    dataset["validation"] = load_dataset("oscar", "unshuffled_deduplicated_no", split=f"train[90%:]")

    column_names = dataset["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    tokenizer = AutoTokenizer.from_pretrained("navjordj/gpt2_no")
    print("2")


    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    print("3")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
            )
        return output

    print("4")

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=True,
        num_proc=6
    )

    import pickle

    print("5")


    with open("tokenized_dataset.pkl", "wb") as f:
        pickle.dump(tokenized_datasets, f)

if __name__ == "__main__":
    freeze_support()

    main()