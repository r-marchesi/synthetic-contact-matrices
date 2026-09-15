# 1. Export your token and set the new cache directory
source .env
export HF_HOME="/storage/DSH/projects/synthetic-contact-matrices/hf_cache"

# 2. Tell Python to download the model to the new cache
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('google/gemma-2-9b-it', token='$HF_TOKEN')"