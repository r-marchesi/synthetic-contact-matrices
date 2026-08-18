import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def generate_matrices(args):
    print(f"Loading tokenizer for {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left" # Required for generation

    print(f"Loading base model {args.model_name} in bfloat16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto" # Automatically uses the GPU
    )

    print(f"Loading LoRA adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    print(f"Reading validation data from {args.validation_file}...")
    with open(args.validation_file, "r") as f:
        validation_data = [json.loads(line) for line in f]

    print(f"Generating contacts for {len(validation_data)} participants...")
    
    with open(args.output_file, "w") as f_out:
        for row in tqdm(validation_data, desc="Generating"):
            # Extract user message and apply chat template
            user_message = next(msg for msg in row["messages"] if msg["role"] == "user")
            formatted_prompt = tokenizer.apply_chat_template(
                [user_message], 
                tokenize=False, 
                add_generation_prompt=True
            )

            # Move inputs to GPU
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    temperature=0.1,      # Low temperature to prevent hallucinations
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Isolate only the newly generated text (ignore the prompt)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Parse the JSON safely (handles potential markdown formatting)
            generated_json = []
            clean_text = raw_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            
            try:
                generated_json = json.loads(clean_text.strip())
            except json.JSONDecodeError:
                print(f"\n[Warning] Failed to parse JSON for part_id {row['part_id']}")
                
            result_row = {
                "part_id": row["part_id"],
                "generated_contacts": generated_json
            }
            f_out.write(json.dumps(result_row) + "\n")

    print(f"\nGeneration complete! Saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Synthetic Matrices with Transformers")
    parser.add_argument("--validation_file", type=str, default="data/splits/validation_set.jsonl")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the trained LoRA folder")
    parser.add_argument("--output_file", type=str, required=True, help="Where to save the JSONL results")
    
    args = parser.parse_args()
    generate_matrices(args)