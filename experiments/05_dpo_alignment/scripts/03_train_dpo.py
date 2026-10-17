import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import PeftModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
EXP04_DIR = os.path.abspath(os.path.join(EXP_DIR, "../04_causal_hierarchy"))
EXP04_SCRIPTS = os.path.join(EXP04_DIR, "scripts")
DATA_DIR = os.path.join(EXP_DIR, "data")
OUT_DIR = os.path.join(EXP_DIR, "checkpoints")

sys.path.append(EXP04_SCRIPTS)
from importlib import import_module
gen04 = import_module("03_generate")


def compute_sequence_logprob(logits, input_ids, labels):
    """
    Computes sum of token log-probabilities and valid token count where labels != -100.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    mask = (shift_labels != -100)

    safe_labels = shift_labels.clone()
    safe_labels[~mask] = 0

    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * mask.float()

    seq_logprob = token_log_probs.sum(dim=-1)
    valid_lengths = mask.sum(dim=-1).clamp(min=1).float()
    return seq_logprob, valid_lengths


def tokenize_pair_side(tokenizer, prompt_str, completion_str, max_seq_length):
    user_msg = [{"role": "user", "content": prompt_str}]
    full_msgs = [
        {"role": "user", "content": prompt_str},
        {"role": "assistant", "content": completion_str},
    ]
    prompt_text = tokenizer.apply_chat_template(user_msg, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_tok = tokenizer(
        full_text, add_special_tokens=False, truncation=True, max_length=max_seq_length
    )
    input_ids = full_tok["input_ids"]
    attention_mask = full_tok["attention_mask"]
    prompt_len = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    return input_ids, attention_mask, labels


class PrecomputedDPODataset(Dataset):
    def __init__(self, pairs_path, model, tokenizer, max_seq_length=2048):
        self.items = []
        raw_pairs = []
        with open(pairs_path, "r") as f:
            for line in f:
                if line.strip():
                    raw_pairs.append(json.loads(line))

        print(f"Pre-computing reference log-probabilities for {len(raw_pairs)} DPO pairs...")
        model.eval()
        device = next(model.parameters()).device

        for p in tqdm(raw_pairs, desc="Ref LogProbs"):
            c_ids, c_mask, c_lab = tokenize_pair_side(tokenizer, p["prompt"], p["chosen"], max_seq_length)
            r_ids, r_mask, r_lab = tokenize_pair_side(tokenizer, p["prompt"], p["rejected"], max_seq_length)

            with torch.no_grad():
                c_in = torch.tensor([c_ids], dtype=torch.long, device=device)
                c_lb = torch.tensor([c_lab], dtype=torch.long, device=device)
                c_logits = model(input_ids=c_in).logits
                ref_c_lp, _ = compute_sequence_logprob(c_logits, c_in, c_lb)

                r_in = torch.tensor([r_ids], dtype=torch.long, device=device)
                r_lb = torch.tensor([r_lab], dtype=torch.long, device=device)
                r_logits = model(input_ids=r_in).logits
                ref_r_lp, _ = compute_sequence_logprob(r_logits, r_in, r_lb)

            self.items.append({
                "chosen_input_ids": c_ids,
                "chosen_attention_mask": c_mask,
                "chosen_labels": c_lab,
                "rejected_input_ids": r_ids,
                "rejected_attention_mask": r_mask,
                "rejected_labels": r_lab,
                "ref_chosen_logps": float(ref_c_lp.item()),
                "ref_rejected_logps": float(ref_r_lp.item()),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class DPODataCollator:
    def __init__(self, pad_token_id):
        self.pad_id = pad_token_id

    def _pad_list(self, seqs, pad_val):
        max_l = max(len(s) for s in seqs)
        return torch.tensor([s + [pad_val] * (max_l - len(s)) for s in seqs], dtype=torch.long)

    def __call__(self, features):
        return {
            "chosen_input_ids": self._pad_list([f["chosen_input_ids"] for f in features], self.pad_id),
            "chosen_attention_mask": self._pad_list([f["chosen_attention_mask"] for f in features], 0),
            "chosen_labels": self._pad_list([f["chosen_labels"] for f in features], -100),
            "rejected_input_ids": self._pad_list([f["rejected_input_ids"] for f in features], self.pad_id),
            "rejected_attention_mask": self._pad_list([f["rejected_attention_mask"] for f in features], 0),
            "rejected_labels": self._pad_list([f["rejected_labels"] for f in features], -100),
            "ref_chosen_logps": torch.tensor([f["ref_chosen_logps"] for f in features], dtype=torch.float32),
            "ref_rejected_logps": torch.tensor([f["ref_rejected_logps"] for f in features], dtype=torch.float32),
        }


class EpidemiologicalDPOTrainer(Trainer):
    def __init__(self, beta=0.1, nll_alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.nll_alpha = nll_alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        c_out = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"],
        )
        pi_c_logps, c_lens = compute_sequence_logprob(
            c_out.logits, inputs["chosen_input_ids"], inputs["chosen_labels"]
        )

        r_out = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"],
        )
        pi_r_logps, _ = compute_sequence_logprob(
            r_out.logits, inputs["rejected_input_ids"], inputs["rejected_labels"]
        )

        ref_c_logps = inputs["ref_chosen_logps"].to(pi_c_logps.device)
        ref_r_logps = inputs["ref_rejected_logps"].to(pi_r_logps.device)

        chosen_rewards = self.beta * (pi_c_logps - ref_c_logps)
        rejected_rewards = self.beta * (pi_r_logps - ref_r_logps)

        dpo_loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
        nll_loss = -(pi_c_logps / c_lens).mean()
        total_loss = dpo_loss + self.nll_alpha * nll_loss

        return (total_loss, c_out) if return_outputs else total_loss


def run_dpo_training(args):
    dpo_out_dir = os.path.join(OUT_DIR, f"dpo_adapter_{args.split_tag}")
    os.makedirs(dpo_out_dir, exist_ok=True)

    sft_adapter_base = os.path.join(EXP04_DIR, "checkpoints", f"adapter_{args.split_tag}")
    sft_ckpt = gen04.find_checkpoint_for_epoch(sft_adapter_base, args.sft_epoch)
    if not sft_ckpt:
        raise FileNotFoundError(f"SFT checkpoint not found for epoch={args.sft_epoch} in {sft_adapter_base}")

    print(f"Loading base model ({args.base_model}) + SFT LoRA adapter ({sft_ckpt})...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    # Load existing SFT LoRA adapter as trainable policy
    model = PeftModel.from_pretrained(base_model, sft_ckpt, is_trainable=True)
    model.print_trainable_parameters()

    pairs_file = os.path.join(DATA_DIR, f"dpo_pairs_{args.split_tag}.jsonl")
    dataset = PrecomputedDPODataset(pairs_file, model, tokenizer, max_seq_length=args.max_seq_len)

    steps_per_epoch = max(1, len(dataset) // (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs

    training_args = TrainingArguments(
        output_dir=dpo_out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(5, int(total_steps * 0.1)),
        num_train_epochs=args.epochs,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        save_strategy="epoch",
        logging_steps=max(1, steps_per_epoch // 5),
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = EpidemiologicalDPOTrainer(
        beta=args.beta,
        nll_alpha=args.nll_alpha,
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=DPODataCollator(pad_token_id=tokenizer.pad_token_id),
    )

    print(f"\n🚀 Starting Epidemiological DPO Training (epochs={args.epochs}, beta={args.beta}, lr={args.lr})...")
    trainer.train()

    final_path = os.path.join(dpo_out_dir, "final_adapter")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n✅ DPO Training complete! Adapter saved to: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--split_tag", type=str, default="100_percent")
    parser.add_argument("--sft_epoch", type=str, default="6")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nll_alpha", type=float, default=0.1)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    args = parser.parse_args()

    run_dpo_training(args)