import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

def format_chat_template(example, tokenizer):
    example["text"] = tokenizer.apply_chat_template(
        example["messages"], 
        tokenize=False, 
        add_generation_prompt=False
    )
    return example

def train_lora(args):
    print(f"Loading dataset from {args.train_file}...")
    dataset = load_dataset("json", data_files={"train": args.train_file})["train"]
    
    print(f"Loading tokenizer for {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = dataset.map(lambda x: format_chat_template(x, tokenizer), num_proc=4)

    print(f"Loading base model {args.model_name} in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Locked-in Expert Hyperparameters for Gemma-2 JSON Generation
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True, 
        optim="adamw_torch",
        report_to="none"
    )

    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        peft_config=lora_config,
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_args
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving final LoRA adapter to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standard bfloat16 LoRA Fine-tuning for Contact Matrices")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the JSONL training split")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it", help="HuggingFace Model ID")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the LoRA adapter")
    
    args = parser.parse_args()
    train_lora(args)