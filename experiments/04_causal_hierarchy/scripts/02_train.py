import os
import json
import torch
import argparse
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(EXP_DIR, "data")
OUT_DIR = os.path.join(EXP_DIR, "checkpoints")


class CompletionOnlyCausalDataset(Dataset):
    """
    Tokenizes ChatML conversations and masks all user prompt tokens with -100
    so loss is computed strictly on the assistant's [PLAN] and [CONTACTS] output.
    """
    def __init__(self, jsonl_path, tokenizer, max_seq_length=2048):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples = []

        with open(jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        user_msg = [{"role": "user", "content": rec["prompt"]}]
        full_msgs = [
            {"role": "user", "content": rec["prompt"]},
            {"role": "assistant", "content": rec["completion"]},
        ]

        # Tokenize prompt up to the assistant generation header
        prompt_text = self.tokenizer.apply_chat_template(
            user_msg, tokenize=False, add_generation_prompt=True
        )
        full_text = self.tokenizer.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False
        )

        prompt_ids = self.tokenizer(
            prompt_text, add_special_tokens=False
        )["input_ids"]
        full_tokenized = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_length,
        )

        input_ids = full_tokenized["input_ids"]
        attention_mask = full_tokenized["attention_mask"]

        # Mask prompt tokens with -100 so loss only trains the assistant completion
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class CompletionDataCollator:
    """Pads input_ids, attention_mask, and labels (-100) dynamically per batch."""
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch_input_ids, batch_mask, batch_labels = [], [], []

        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch_input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            batch_mask.append(f["attention_mask"] + [0] * pad_len)
            batch_labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def train_model(args):
    split_out_dir = os.path.join(OUT_DIR, f"adapter_{args.split_tag}")
    os.makedirs(split_out_dir, exist_ok=True)

    print(f"Loading tokenizer & model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # Recommended for Gemma-2 stability
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_file = os.path.join(DATA_DIR, f"train_causal_{args.split_tag}.jsonl")
    val_file = os.path.join(DATA_DIR, "val_causal.jsonl")

    train_dataset = CompletionOnlyCausalDataset(train_file, tokenizer, max_seq_length=args.max_seq_len)
    val_dataset = CompletionOnlyCausalDataset(val_file, tokenizer, max_seq_length=args.max_seq_len)

    steps_per_epoch = max(1, len(train_dataset) // (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(5, int(total_steps * 0.06))

    training_args = TrainingArguments(
        output_dir=split_out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        num_train_epochs=args.epochs,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=max(1, steps_per_epoch // 4),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        data_collator=CompletionDataCollator(pad_token_id=tokenizer.pad_token_id),
    )

    print(f"\n🚀 Starting Completion-Only LoRA Training ({args.epochs} epochs, Cosine LR)...")
    trainer.train()

    final_path = os.path.join(split_out_dir, "final_adapter")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"✅ Training complete! Saved to: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--split_tag", type=str, default="100_percent")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    args = parser.parse_args()

    train_model(args)