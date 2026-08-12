import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

def train_lora(args):
    print(f"Loading dataset from {args.train_file}...")
    dataset = load_dataset("json", data_files={"train": args.train_file})["train"]
    
    print(f"Loading tokenizer for {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Setup padding token and explicitly set right-padding for Causal LM
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def preprocess_function(examples):
        # Because batched=True, examples["messages"] is a list of lists
        texts = [
            tokenizer.apply_chat_template(
                msg, 
                tokenize=False, 
                add_generation_prompt=False
            ) for msg in examples["messages"]
        ]
        
        # Tokenize without padding (the collator pads dynamically per batch)
        return tokenizer(
            texts,
            truncation=True,
            max_length=2048,
            padding=False
        )

    print("Tokenizing dataset...")
    dataset = dataset.map(
        preprocess_function, 
        batched=True, 
        num_proc=4, 
        remove_columns=dataset.column_names
    )

    print(f"Loading base model {args.model_name} in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    print("Applying LoRA adapter to model...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,            # Lowered from 4 to 1
        gradient_accumulation_steps=16,           # Raised from 4 to 16
        gradient_checkpointing=True,              # Added this line
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

    print("Initializing standard Trainer...")
    # mlm=False tells the collator to automatically build causal 'labels' padded with -100
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving final LoRA adapter to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standard bfloat16 LoRA Fine-tuning")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the JSONL training split")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it", help="HuggingFace Model ID")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the LoRA adapter")
    
    args = parser.parse_args()
    train_lora(args)