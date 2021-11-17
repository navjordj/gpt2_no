from transformers import (
    AutoTokenizer
)

tokenizer = AutoTokenizer.from_pretrained("gpt2_no/")

tokenizer.push_to_hub("navjord/gpt2_no")
