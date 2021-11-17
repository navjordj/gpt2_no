from transformers import GPT2LMHeadModel

mdl_path = "gpt2_no"

pt_model = GPT2LMHeadModel.from_pretrained(mdl_path, from_flax=True)
pt_model.save_pretrained(mdl_path)

