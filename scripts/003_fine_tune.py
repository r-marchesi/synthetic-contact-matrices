import os
import sys
import json
import random
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# Ensure local script imports work regardless of execution root
sys.path.append(os.path.dirname(__file__))
from utils_prompting import format_conversation

DEFAULT_CONFIG = {
    "experiment_name": "exp_01_baseline",
    "prompt_style": "json_strict",
    "model_name": "google/gemma-2-9b-it",
    "learning_rate": 1e-4,
    "num_train_epochs": 3,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "max_seq_length": 8192,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "warmup_ratio": 0.1,
    "shuffle_contacts": True
}

CONTACT_FIELDS = [
    "Contact Age",
    "Contact Gender",
    "Contact Frequency",
    "Physical Contact",
    "Distance from Home",
    "Relationship to Participant",
    "Contact Setting",
    "Location of Contact"
]

class DynamicContactDataset(Dataset):
    """
    Applies on-the-fly permutation augmentation to contacts on every epoch,
    preventing the autoregressive model from memorizing fixed sequence orders.
    """
    def __init__(self, records, tokenizer, prompt_style, max_seq_length, shuffle_contacts=True):
        self.records = records
        self.tokenizer = tokenizer
        self.prompt_style = prompt_style
        self.max_seq_length = max_seq_length
        self.shuffle_contacts = shuffle_contacts

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        demos = item["demos"]
        contacts = list(item["contacts"])

        # Dynamic augmentation: permutation invariance
        if self.shuffle_contacts and len(contacts) > 1:
            random.shuffle(contacts)

        messages = format_conversation(demos, contacts, prompt_style=self.prompt_style)
        formatted_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        tokenized = self.tokenizer(
            formatted_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors=None
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"]
        }

def load_and_package_data(master_csv, train_ids_path):
    print(f"Loading master data: {master_csv}")
    master_df = pd.read_csv(master_csv, low_memory=False)
    
    print(f"Filtering by training cohort: {train_ids_path}")
    train_ids_df = pd.read_csv(train_ids_path)
    train_ids_set = set(train_ids_df["part_id"].astype(str).unique())

    subset_df = master_df[master_df["part_id"].astype(str).isin(train_ids_set)].copy()
    grouped = subset_df.groupby("part_id")

    records = []
    for part_id, group in grouped:
        first_row = group.iloc[0]
        demos = {
            "age": first_row.get("Participant Age", "Unknown"),
            "gender": first_row.get("Participant Gender", "Unknown"),
            "hh_size": first_row.get("Household Size", "Unknown"),
            "occupation": first_row.get("Occupation", "Unknown"),
            "work": first_row.get("Attends Work", "Unknown"),
            "school": first_row.get("Attends School", "Unknown"),
            "education": first_row.get("Educational Attainment", "Unknown"),
            "region": first_row.get("Region", "Unknown"),
            "income": first_row.get("Household Monthly Net Income", "Unknown")
        }

        contacts = []
        for _, row in group.iterrows():
            # Check for non-null/valid contact row
            cont_id = row.get("cont_id")
            if pd.isna(cont_id) or str(cont_id).strip().lower() in ["", "unknown", "nan"]:
                continue

            c_dict = {}
            for field in CONTACT_FIELDS:
                val = row.get(field, "Unknown")
                c_dict[field] = "Unknown" if pd.isna(val) else val
            contacts.append(c_dict)

        records.append({
            "part_id": str(part_id),
            "demos": demos,
            "contacts": contacts
        })

    print(f"Loaded {len(records)} participants for training.")
    return records

def train(args):
    # Load and merge configuration
    config = dict(DEFAULT_CONFIG)
    if args.config and os.path.exists(args.config):
        print(f"Loading experiment configuration from: {args.config}")
        with open(args.config, "r") as f:
            user_cfg = json.load(f)
            config.update(user_cfg)
    else:
        print("Using default baseline experiment configuration.")

    # Infer split name from IDs file if not provided
    split_name = os.path.splitext(os.path.basename(args.train_ids))[0].replace("train_ids_", "")
    output_dir = args.output_dir or os.path.join("data/adapters", f"{config['experiment_name']}_{split_name}")
    os.makedirs(output_dir, exist_ok=True)

    # Save active configuration to adapter directory for strict provenance
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Tokenizer setup
    print(f"Loading tokenizer: {config['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare datasets
    records = load_and_package_data(args.master_csv, args.train_ids)
    train_dataset = DynamicContactDataset(
        records=records,
        tokenizer=tokenizer,
        prompt_style=config["prompt_style"],
        max_seq_length=config["max_seq_length"],
        shuffle_contacts=config["shuffle_contacts"]
    )

    # Base model setup
    print(f"Loading base model {config['model_name']} in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        dtype=torch.bfloat16,
        device_map="auto"
    )

    # LoRA setup
    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dynamic step and warmup calculations
    grad_accum = config["gradient_accumulation_steps"]
    batch_size = config["per_device_train_batch_size"]
    epochs = config["num_train_epochs"]
    
    steps_per_epoch = max(1, len(train_dataset) // (batch_size * grad_accum))
    total_steps = steps_per_epoch * epochs
    dynamic_warmup = max(5, int(total_steps * config["warmup_ratio"]))
    print(f"Total training steps: {total_steps} | Dynamic warmup steps: {dynamic_warmup}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        learning_rate=config["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_steps=dynamic_warmup,
        num_train_epochs=epochs,
        logging_steps=max(1, steps_per_epoch // 4),
        save_strategy="epoch",
        bf16=True,
        optim="adamw_torch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    print("Beginning fine-tuning...")
    trainer.train()

    print(f"Saving final adapter and tokenizer to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Fine-tuning completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modular LoRA Fine-Tuning Sweep Engine")
    parser.add_argument("--config", type=str, default=None, help="Path to experiment JSON config")
    parser.add_argument("--master_csv", type=str, default="data/processed/master_data.csv", help="Master flat data CSV")
    parser.add_argument("--train_ids", type=str, default="data/splits/train_ids_100_percent.csv", help="Path to train cohort IDs CSV")
    parser.add_argument("--output_dir", type=str, default=None, help="Explicit adapter output directory")
    
    args = parser.parse_args()
    train(args)