import os
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# --- PATH RESOLUTION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(EXP_DIR, "data")
OUT_DIR = os.path.join(EXP_DIR, "checkpoints")

def train_model(model_id="google/gemma-2-9b-it", epochs=25, batch_size=2, lr=5e-5, warmup_ratio=0.05):
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print(f"Loading tokenizer and model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # --- LORA CONFIGURATION ---
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- DATASET LOADING & TOKENIZATION ---
    train_file = os.path.join(DATA_DIR, "train_dense.jsonl")
    val_file = os.path.join(DATA_DIR, "val_dense.jsonl")
    dataset = load_dataset("json", data_files={"train": train_file, "val": val_file})
    
    def format_and_tokenize(example):
        text = example["prompt"] + example["completion"] + tokenizer.eos_token
        return tokenizer(text, truncation=True, max_length=1024, padding=False)
        
    tokenized_dataset = dataset.map(
        format_and_tokenize, 
        remove_columns=dataset["train"].column_names
    )

    # --- WARMUP & STEP CALCULATION ---
    # Compensating for the halved batch size to maintain effective batch size
    grad_accum = 8 
    steps_per_epoch = max(1, len(tokenized_dataset["train"]) // (batch_size * grad_accum))
    total_steps = steps_per_epoch * epochs
    dynamic_warmup = max(5, int(total_steps * warmup_ratio))
    
    print(f"Total Steps: {total_steps} | Warmup Steps: {dynamic_warmup}")

    # --- CORE TRANSFORMERS TRAINING ARGUMENTS ---
    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=dynamic_warmup,      
        num_train_epochs=epochs,
        bf16=True,
        gradient_checkpointing=True,      # Prevents OOM by recomputing activations
        optim="adamw_torch",              # Reverted to standard 32-bit AdamW
        eval_strategy="epoch",            
        save_strategy="epoch",
        logging_steps=20,
        report_to="none",
    )

    # --- STANDARD TRAINER ---
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    print(f"\n🚀 Starting {epochs}-Epoch Training Run...")
    trainer.train()
    
    final_path = os.path.join(OUT_DIR, "final_adapter")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n✅ Training complete! Adapters saved to {OUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=2) # Defaulting to safe batch size
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()
    
    train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)