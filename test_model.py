from transformers import AutoTokenizer, AutoModelWithLMHead, FlaxGPT2LMHeadModel, TextGenerationPipeline, pipeline, GPT2LMHeadModel
tokenizer = AutoTokenizer.from_pretrained("navjordj/gpt2_no")

model = GPT2LMHeadModel.from_pretrained("navjordj/gpt2_no", from_flax=True)
tokenizer = AutoTokenizer.from_pretrained("navjordj/gpt2_no")


pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

res = pipe("I dag klokken ")
print(res)
