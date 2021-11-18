from transformers import AutoTokenizer, AutoModelWithLMHead, FlaxGPT2LMHeadModel, TextGenerationPipeline, pipeline, GPT2LMHeadModel
tokenizer = AutoTokenizer.from_pretrained("navjordj/gpt2_no")

model = GPT2LMHeadModel.from_pretrained("navjordj/gpt2_no", from_flax=True, revision="c1dfa0b823ee94df6bc51ef220c367424b76ac63")
tokenizer = AutoTokenizer.from_pretrained("navjordj/gpt2_no")


pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

res = pipe("I dag klokken ")
print(res)
