import json
import os
import glob
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys

sys.path.append(os.path.dirname(__file__))
from utils_prompting import format_conversation

CONTACT_MAP = {
    'cnt_age_exact': 'Contact Age',
    'cnt_gender': 'Contact Gender',
    'frequency_multi': 'Contact Frequency',
    'phys_contact': 'Physical Contact',
    'distance': 'Distance during Contact', 
    'relationship': 'Relationship to Participant',
    'setting': 'Contact Setting',
    'location_multi': 'Location of Contact'
}

def find_checkpoint_for_epoch(base_dir, target_epoch):
    """Parses trainer_state.json across checkpoints to find the exact step for the epoch."""
    checkpoints = glob.glob(os.path.join(base_dir, "checkpoint-*"))
    for ckpt in checkpoints:
        state_file = os.path.join(ckpt, "trainer_state.json")
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                state = json.load(f)
                # Trainer sometimes saves floats like 2.0001
                if round(state.get("epoch", 0)) == target_epoch:
                    return ckpt
    return None

def load_validation_records(master_csv, val_ids_path, prompt_style):
    print(f"Loading master data: {master_csv}")
    master_df = pd.read_csv(master_csv, low_memory=False)
    
    val_ids_df = pd.read_csv(val_ids_path)
    val_ids_set = set(val_ids_df["part_id"].astype(str).unique())

    subset_df = master_df[master_df["part_id"].astype(str).isin(val_ids_set)].copy()
    grouped = subset_df.groupby("part_id")

    records = []
    for part_id, group in grouped:
        first_row = group.iloc[0]
        demos = {
            "age": first_row.get("part_age_exact", "Unknown"),
            "gender": first_row.get("part_gender", "Unknown"),
            "hh_size": first_row.get("hh_size", "Unknown"),
            "occupation": first_row.get("occupation", "Unknown")
        }

        contacts = []
        for _, row in group.iterrows():
            cont_id = row.get("cont_id")
            if pd.isna(cont_id) or str(cont_id).strip().lower() in ["", "unknown", "nan"]:
                continue

            c_dict = {}
            for raw_col, formatted_key in CONTACT_MAP.items():
                val = row.get(raw_col, "Unknown")
                c_dict[formatted_key] = "Unknown" if pd.isna(val) else val
            contacts.append(c_dict)

        messages = format_conversation(demos, contacts, prompt_style=prompt_style)
        records.append({
            "part_id": str(part_id),
            "user_prompt": next(m["content"] for m in messages if m["role"] == "user"),
            "target_response": next(m["content"] for m in messages if m["role"] == "assistant")
        })

    return records

def generate_matrices(args):
    with open(args.config, "r") as f:
        config = json.load(f)
        
    exp_name = config.get("experiment_name")
    prompt_style = config.get("prompt_style")
    model_name = config.get("model_name")
    
    adapter_base = os.path.join("data/adapters", f"{exp_name}_100_percent")
    checkpoint_path = find_checkpoint_for_epoch(adapter_base, args.target_epoch)
    
    if not checkpoint_path:
        print(f"Error: Could not find checkpoint for epoch {args.target_epoch} in {adapter_base}")
        sys.exit(1)

    print(f"==================================================")
    print(f"Experiment: {exp_name} | Style: {prompt_style} | Epoch: {args.target_epoch}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"==================================================")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left" 

    print("Loading base model in bfloat16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    print(f"Merging LoRA weights from {checkpoint_path}...")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model = model.merge_and_unload() 
    model.eval()

    records = load_validation_records(args.master_csv, args.val_ids, prompt_style)

    print("Analyzing ground-truth data to calculate optimal max_new_tokens...")
    max_target_tokens = 0
    for r in records:
        tokens = tokenizer.encode(r["target_response"], add_special_tokens=False)
        if len(tokens) > max_target_tokens:
            max_target_tokens = len(tokens)
            
    dynamic_max_tokens = int(max_target_tokens * 1.20)
    print(f"--> Longest target sequence is {max_target_tokens} tokens.")
    print(f"--> Setting max_new_tokens to {dynamic_max_tokens} (with 20% safety buffer).")

    stop_token_ids = [tokenizer.eos_token_id]
    if "<end_of_turn>" in tokenizer.vocab:
        stop_token_ids.append(tokenizer.vocab["<end_of_turn>"])

    output_file = os.path.join(args.output_dir, f"{exp_name}_epoch_{args.target_epoch}_results.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating contacts for {len(records)} participants...")
    with open(output_file, "w") as f_out:
        for r in tqdm(records, desc="Generating"):
            formatted_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": r["user_prompt"]}], 
                tokenize=False, 
                add_generation_prompt=True
            )

            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=dynamic_max_tokens,
                    temperature=0.6,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    do_sample=True,
                    eos_token_id=stop_token_ids,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )

            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Save the raw text output. Parsing logic will move to script 05 
            # so we don't break generation if a format deviates.
            result_row = {
                "part_id": r["part_id"],
                "prompt_style": prompt_style,
                "raw_generated_text": raw_text
            }
            f_out.write(json.dumps(result_row) + "\n")
            f_out.flush()

    print(f"\nGeneration complete! Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--target_epoch", type=int, required=True)
    parser.add_argument("--val_ids", type=str, default="data/splits/val_ids.csv")
    parser.add_argument("--master_csv", type=str, default="data/processed/master_data.csv")
    parser.add_argument("--output_dir", type=str, default="data/results")
    
    args = parser.parse_args()
    generate_matrices(args)