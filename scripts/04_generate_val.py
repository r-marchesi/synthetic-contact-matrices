import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def generate_matrices(args):
    print(f"Loading tokenizer for {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left" 

    print(f"Loading base model {args.model_name} in bfloat16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print(f"Loading LoRA adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    
    print("Merging LoRA weights into base model for faster inference...")
    model = model.merge_and_unload() 
    model.eval()

    print(f"Reading validation data from {args.validation_file}...")
    with open(args.validation_file, "r") as f:
        validation_data = [json.loads(line) for line in f]

    # =====================================================================
    # DATA-DRIVEN TOKEN MATH: Find the longest real matrix in the data
    # =====================================================================
    print("Analyzing ground-truth data to calculate optimal max_new_tokens...")
    max_target_tokens = 0
    
    for row in validation_data:
        assistant_msg = next(msg for msg in row["messages"] if msg["role"] == "assistant")
        # Tokenize the actual JSON output to count its true length
        tokens = tokenizer.encode(assistant_msg["content"], add_special_tokens=False)
        if len(tokens) > max_target_tokens:
            max_target_tokens = len(tokens)
            
    # Add a 20% safety threshold to the longest real matrix
    dynamic_max_tokens = int(max_target_tokens * 1.20)
    print(f"--> Longest real matrix is {max_target_tokens} tokens.")
    print(f"--> Setting max_new_tokens to {dynamic_max_tokens} (with 20% safety buffer).")
    # =====================================================================

    stop_token_ids = [tokenizer.eos_token_id]
    if "<end_of_turn>" in tokenizer.vocab:
        stop_token_ids.append(tokenizer.vocab["<end_of_turn>"])

    print(f"Generating contacts for {len(validation_data)} participants...")
    
    with open(args.output_file, "w") as f_out:
        for row in tqdm(validation_data, desc="Generating"):
            user_message = next(msg for msg in row["messages"] if msg["role"] == "user")
            formatted_prompt = tokenizer.apply_chat_template(
                [user_message], 
                tokenize=False, 
                add_generation_prompt=True
            )

            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=dynamic_max_tokens,  # <-- Using your data-driven limit!
                    temperature=0.1,      
                    do_sample=True,
                    eos_token_id=stop_token_ids,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )

            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            clean_text = raw_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            
            clean_text = clean_text.strip()
            
            # --- AUTO-FIX TRUNCATED JSON ---
            if not clean_text.endswith("]"):
                last_brace = clean_text.rfind("}")
                if last_brace != -1:
                    clean_text = clean_text[:last_brace+1] + "\n]"
                else:
                    clean_text = "[]"
            # -------------------------------
            
            try:
                generated_json = json.loads(clean_text)
            except json.JSONDecodeError:
                generated_json = []
                
            result_row = {
                "part_id": row["part_id"],
                "generated_contacts": generated_json
            }
            f_out.write(json.dumps(result_row) + "\n")
            f_out.flush()

    print(f"\nGeneration complete! Saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Synthetic Matrices with Transformers")
    parser.add_argument("--validation_file", type=str, default="data/splits/validation_set.jsonl")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the trained LoRA folder")
    parser.add_argument("--output_file", type=str, required=True, help="Where to save the JSONL results")
    
    args = parser.parse_args()
    generate_matrices(args)