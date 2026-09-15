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

sys.path.append(os.path.dirname(__file__))
from utils_prompting import format_conversation

DEFAULT_CONFIG = {
    "experiment_name": "exp_01_baseline",
    "prompt_style": "json_strict",
    "model_name": "google/gemma-2-9b-it",
    "learning_rate": 1e-4,
    "num_train_epochs": 10,
    "lr_scheduler_type": "constant_with_warmup",
    "save_strategy": "epoch",
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

# Translates raw CSV columns to the natural language keys expected by utils_prompting.py
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

class DynamicContactDataset(Dataset):
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

        if self.shuffle_contacts and len(contacts) > 1:
            random.shuffle(contacts)

        messages = format_conversation(demos, contacts, prompt_style=self.prompt_style)
        formatted_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        tokenized = self.tokenizer(
            formatted_text, truncation=True, max_length=self.max_seq_length, padding=False
        )
        return {"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"]}

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
        
        # Read raw demographics
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

            # Map raw contact columns to natural language dictionary
            c_dict = {}
            for raw_col, formatted_key in CONTACT_MAP.items():
                val = row.get(raw_col, "Unknown")
                c_dict[formatted_key] = "Unknown" if pd.isna(val) else val
            contacts.append(c_dict)

        records.append({"part_id": str(part_id), "demos": demos, "contacts": contacts})

    print(f"Loaded {len(records)} participants for training.")
    return records

def train(args):
    config = dict(DEFAULT_CONFIG)
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config.update(json.load(f))

    split_name = os.path.splitext(os.path.basename(args.train_ids))[0].replace("train_ids_", "")
    output_dir = args.output_dir or os.path.join("data/adapters", f"{config['experiment_name']}_{split_name}")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    records = load_and_package_data(args.master_csv, args.train_ids)
    train_dataset = DynamicContactDataset(
        records=records,
        tokenizer=tokenizer,
        prompt_style=config["prompt_style"],
        max_seq_length=config["max_seq_length"],
        shuffle_contacts=config["shuffle_contacts"]
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"], dtype=torch.bfloat16, device_map="auto"
    )

    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    grad_accum = config["gradient_accumulation_steps"]
    batch_size = config["per_device_train_batch_size"]
    epochs = config["num_train_epochs"]
    
    steps_per_epoch = max(1, len(train_dataset) // (batch_size * grad_accum))
    total_steps = steps_per_epoch * epochs
    dynamic_warmup = max(5, int(total_steps * config["warmup_ratio"]))

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_steps=dynamic_warmup,
        num_train_epochs=epochs,
        logging_steps=max(1, steps_per_epoch // 4),
        save_strategy=config["save_strategy"],
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

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--master_csv", type=str, default="data/processed/master_data.csv")
    parser.add_argument("--train_ids", type=str, default="data/splits/train_ids_100_percent.csv")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    train(args)