from transformers import AutoModel

from huggingface_hub import Repository, get_full_repo_name

model_name = "gpt2_no"

repo_name = get_full_repo_name(model_name)
print(repo_name)

output_dir = "hub_gpt2_no"

repo = Repository(output_dir, clone_from=repo_name)

repo.push_to_hub(commit_message="initial commit", blocking=False)
