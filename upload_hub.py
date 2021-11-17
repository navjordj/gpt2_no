import sys

from transformers import AutoModel

from huggingface_hub import Repository, get_full_repo_name

model_name = "gpt2_no"

repo_name = get_full_repo_name(model_name)
print(repo_name)

output_dir = "gpt2_no"

repo = Repository(output_dir, clone_from=repo_name)
repo.git_pull()

if len(sys.argv) > 1:
    commit_message = sys.argv[1]
else:
    commit_message = f"{model_name} commit"


repo.push_to_hub(commit_message=commit_message, blocking=False)
