import os
import json
import argparse
import torch
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


class CompletionOnlyContactDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_seq_length=2048):
        self.samples = []
        with open(jsonl_path, "r") as f:
            raw_lines = [json.loads(line) for line in f if line.strip()]

        for item in raw_lines:
            user_msg = [{"role": "user", "content": item["prompt"]}]
            full_msgs = [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["completion"]},
            ]

            prompt_text = tokenizer.apply_chat_template(
                user_msg, tokenize=False, add_generation_prompt=True
            )
            full_text = tokenizer.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False
            )

            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_tok = tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_seq_length,
            )
            input_ids = full_tok["input_ids"]
            attention_mask = full_tok["attention_mask"]

            prompt_len = min(len(prompt_ids), len(input_ids))
            labels = [-100] * prompt_len + input_ids[prompt_len:]

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class CompletionOnlyCollator:
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


def train(args):
    train_file = os.path.join(DATA_DIR, f"train_causal_{args.split_tag}.jsonl")
    adapter_out_dir = os.path.join(OUT_DIR, f"adapter_{args.split_tag}")
    os.makedirs(adapter_out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = CompletionOnlyContactDataset(
        train_file, tokenizer, max_seq_length=args.max_seq_length
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
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
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    steps_per_epoch = max(1, len(dataset) // (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs

    training_args = TrainingArguments(
        output_dir=adapter_out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_steps=max(10, int(total_steps * 0.06)),
        num_train_epochs=args.epochs,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        save_strategy="epoch",
        save_total_limit=args.epochs,
        logging_steps=max(1, steps_per_epoch // 5),
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=CompletionOnlyCollator(pad_token_id=tokenizer.pad_token_id),
    )

    print(f"🚀 Starting Exp 07 Semantic Causal SFT ({args.epochs} epochs)...")
    trainer.train()

    final_dir = os.path.join(adapter_out_dir, "final_adapter")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"✅ Training complete. Final adapter saved to {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--split_tag", type=str, default="100_percent")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    args = parser.parse_args()

    train(args)