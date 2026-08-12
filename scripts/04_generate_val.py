import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

def generate_matrices(args):
    print(f"Loading tokenizer for {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    print(f"Reading validation data from {args.validation_file}...")
    with open(args.validation_file, "r") as f:
        validation_data = [json.loads(line) for line in f]

    # Extract user prompts and format them using the Gemma 2 chat template
    prompts = []
    for row in validation_data:
        # Extract just the user's message
        user_message = next(msg for msg in row["messages"] if msg["role"] == "user")
        
        # Apply the chat template and add the "<start_of_turn>model\n" prompt
        formatted_prompt = tokenizer.apply_chat_template(
            [user_message], 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompts.append(formatted_prompt)

    # Define the exact JSON schema based on the training data
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "Contact Age": {"type": "integer"},
                "Contact Gender": {"type": "string"},
                "Contact Frequency": {"type": "number"},
                "Physical Contact": {"type": "boolean"},
                "Distance from Home": {"type": "string"},
                "Relationship to Participant": {"type": "string"},
                "Contact Setting": {"type": "string"},
                "Location of Contact": {"type": "string"}
            },
            "required": [
                "Contact Age", "Contact Gender", "Contact Frequency", 
                "Physical Contact", "Distance from Home", 
                "Relationship to Participant", "Contact Setting", "Location of Contact"
            ]
        }
    }

    # Initialize vLLM with LoRA support enabled
    print("Initializing vLLM Engine...")
    llm = LLM(
        model=args.model_name,
        enable_lora=True,
        max_lora_rank=16,
        max_model_len=4096,
        gpu_memory_utilization=0.90, # Gives vLLM plenty of breathing room
        dtype="bfloat16"
    )

    # Set up guided decoding
    sampling_params = SamplingParams(
        temperature=0.1,  # Low temperature to keep the model grounded
        max_tokens=4096,
        guided_json=json.dumps(schema)
    )

    # Generate the responses using the specific LoRA adapter
    print(f"Generating contacts using adapter: {args.adapter_path}")
    outputs = llm.generate(
        prompts, 
        sampling_params, 
        lora_request=LoRARequest("contact_adapter", 1, args.adapter_path)
    )

    # Parse and save the results
    print(f"Saving generated matrices to {args.output_file}...")
    with open(args.output_file, "w") as f:
        for i, output in enumerate(outputs):
            raw_text = output.outputs[0].text
            
            try:
                # Parse the raw text back into a Python list
                generated_json = json.loads(raw_text)
            except json.JSONDecodeError:
                generated_json = [] # Fallback, though guided decoding makes this practically impossible
                
            result_row = {
                "part_id": validation_data[i]["part_id"],
                "generated_contacts": generated_json
            }
            f.write(json.dumps(result_row) + "\n")

    print("Generation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Synthetic Matrices with vLLM")
    parser.add_argument("--validation_file", type=str, default="data/splits/validation_set.jsonl")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the trained LoRA folder")
    parser.add_argument("--output_file", type=str, required=True, help="Where to save the JSONL results")
    
    args = parser.parse_args()
    generate_matrices(args)