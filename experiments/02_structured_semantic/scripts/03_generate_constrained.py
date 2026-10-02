import os
import json
import argparse
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

# --- PATH RESOLUTION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(EXP_DIR, "data")
CHKPT_DIR = os.path.join(EXP_DIR, "checkpoints")
OUT_DIR = os.path.join(EXP_DIR, "generated")

def load_model_and_tokenizer(base_model_id, lora_path):
    print(f"\nLoading Base Model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"Applying LoRA Adapter from: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    # Merge for faster inference
    model = model.merge_and_unload()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # Left padding for batched generation
    
    return model, tokenizer

def generate_unconstrained(model_id, adapter_epoch, num_samples):
    """
    Standard generation. Our semantic format is structured enough that the model 
    should implicitly learn the grammar without forced logits masking.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    
    adapter_path = os.path.join(CHKPT_DIR, f"checkpoint-{adapter_epoch}")
    if not os.path.exists(adapter_path):
        adapter_path = os.path.join(CHKPT_DIR, "final_adapter")
        print(f"Warning: Checkpoint {adapter_epoch} not found. Using {adapter_path}")

    model, tokenizer = load_model_and_tokenizer(model_id, adapter_path)
    
    val_file = os.path.join(DATA_DIR, "val_dense.jsonl")
    df_val = pd.read_json(val_file, lines=True)
    
    if num_samples > 0:
        subset = df_val.head(num_samples)
    else:
        subset = df_val
        
    prompts = subset["prompt"].tolist()
    part_ids = subset["part_id"].tolist()

    total_records = len(subset)
    print(f"\n🚀 Generating {total_records} records for Epoch {adapter_epoch}...")
    
    results = []
    
    # Process in chunks to speed up generation on the H200
    batch_size = 16 
    
    model.eval()
    for i in range(0, total_records, batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_ids = part_ids[i:i+batch_size]
        
        print(f"Processing batch {i//batch_size + 1}/{(total_records + batch_size - 1)//batch_size} (Records {i} to {i+len(batch_prompts)})")
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,      # Slight temperature for natural variation
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the newly generated tokens
        input_lengths = [len(inp) for inp in inputs.input_ids]
        for j, output_seq in enumerate(outputs):
            gen_tokens = output_seq[input_lengths[j]:]
            completion_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            
            results.append({
                "part_id": batch_ids[j],
                "prompt": batch_prompts[j].strip(),
                "generated_completion": completion_text.strip()
            })

    out_file = os.path.join(OUT_DIR, f"generated_epoch_{adapter_epoch}.jsonl")
    pd.DataFrame(results).to_json(out_file, orient="records", lines=True)
    print(f"\n✅ Generation complete! Saved {total_records} records to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--epoch", type=str, default="25")
    parser.add_argument("--samples", type=int, default=-1)
    args = parser.parse_args()
    
    generate_unconstrained(args.base_model, args.epoch, args.samples)