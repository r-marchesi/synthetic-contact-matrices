import os
import re
import glob
import json
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(EXP_DIR, "data")
CHKPT_DIR = os.path.join(EXP_DIR, "checkpoints")
OUT_DIR = os.path.join(EXP_DIR, "generated")

SETTING_ORDER = ["Home", "Work", "School", "Other"]
PLAN_PARSE_RE = re.compile(
    r"Home:\s*(\d+)\s*\|\s*Work:\s*(\d+)\s*\|\s*School:\s*(\d+)\s*\|\s*Other:\s*(\d+)",
    re.IGNORECASE,
)


def find_checkpoint_for_epoch(adapter_base, target_epoch):
    """Finds the exact checkpoint folder matching target_epoch via trainer_state.json."""
    if str(target_epoch).lower() == "final":
        final_dir = os.path.join(adapter_base, "final_adapter")
        return final_dir if os.path.exists(final_dir) else None

    target_epoch_int = int(target_epoch)
    checkpoints = glob.glob(os.path.join(adapter_base, "checkpoint-*"))
    for ckpt in sorted(checkpoints):
        state_file = os.path.join(ckpt, "trainer_state.json")
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                state = json.load(f)
                if round(state.get("epoch", 0)) == target_epoch_int:
                    return ckpt

    # Fallback to final_adapter if exact epoch not found
    fallback = os.path.join(adapter_base, "final_adapter")
    if os.path.exists(fallback):
        print(f"Warning: Epoch {target_epoch} not found. Falling back to {fallback}")
        return fallback
    return None


def load_model_and_tokenizer(base_model_id, lora_path):
    print(f"Loading base model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    print(f"Merging LoRA adapter from: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    model = model.merge_and_unload()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def get_newline_token_ids(tokenizer):
    """Collects token IDs corresponding to newline characters for single-line stopping."""
    stop_ids = {tokenizer.eos_token_id}
    if "<end_of_turn>" in tokenizer.vocab:
        stop_ids.add(tokenizer.vocab["<end_of_turn>"])
    for tok_str in ["\n", "\r\n"]:
        ids = tokenizer.encode(tok_str, add_special_tokens=False)
        if len(ids) == 1:
            stop_ids.add(ids[0])
    return list(stop_ids)


def parse_planned_schedule(plan_text, max_contacts=35):
    """
    Extracts (Home, Work, School, Other) counts from the generated [PLAN] line
    and returns a canonical list of settings to generate, e.g. ['Home', 'Home', 'Work'].
    """
    m = PLAN_PARSE_RE.search(plan_text)
    if not m:
        return [], "[PLAN]\nTotal: 0 | Home: 0 | Work: 0 | School: 0 | Other: 0"

    counts = {
        "Home": min(15, int(m.group(1))),
        "Work": min(25, int(m.group(2))),
        "School": min(25, int(m.group(3))),
        "Other": min(20, int(m.group(4))),
    }
    schedule = []
    for s in SETTING_ORDER:
        schedule.extend([s] * counts[s])

    # Cap total contacts at max_contacts while preserving setting order
    schedule = schedule[:max_contacts]
    reconciled_counts = {s: schedule.count(s) for s in SETTING_ORDER}
    total = len(schedule)

    clean_plan_header = (
        f"[PLAN]\n"
        f"Total: {total} | Home: {reconciled_counts['Home']} | "
        f"Work: {reconciled_counts['Work']} | School: {reconciled_counts['School']} | "
        f"Other: {reconciled_counts['Other']}"
    )
    return schedule, clean_plan_header


def generate_batch_plan_guided(model, tokenizer, batch_prompts, args, line_stop_ids):
    """
    Two-phase generation:
      Phase 1: Sample the [PLAN] line for all participants in the batch.
      Phase 2: Step-by-step prefix-force '- Setting: <S> | Location:' for each planned contact.
    """
    # Phase 1: Format ChatML prompt + '[PLAN]\nTotal:' prefix
    chat_prefixes = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        + "[PLAN]\nTotal:"
        for p in batch_prompts
    ]

    inputs = tokenizer(chat_prefixes, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        plan_outputs = model.generate(
            **inputs,
            max_new_tokens=36,
            do_sample=True,
            temperature=args.temperature,
            min_p=args.min_p,
            top_p=args.top_p,
            eos_token_id=line_stop_ids,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

    schedules = []
    running_completions = []
    base_chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in batch_prompts
    ]

    in_lens = [len(x) for x in inputs.input_ids]
    for i, out_seq in enumerate(plan_outputs):
        gen_Part = tokenizer.decode(out_seq[in_lens[i]:], skip_special_tokens=True).split("\n")[0]
        raw_plan_line = "Total:" + gen_Part
        sched, clean_header = parse_planned_schedule(raw_plan_line, max_contacts=args.max_contacts)
        schedules.append(sched)
        if len(sched) == 0:
            running_completions.append(f"{clean_header}\n[CONTACTS]\n- None")
        else:
            running_completions.append(f"{clean_header}\n[CONTACTS]")

    # Phase 2: Iteratively generate contact lines for active participants in the batch
    max_steps = max((len(s) for s in schedules), default=0)
    for step_idx in range(max_steps):
        active_indices = [i for i, s in enumerate(schedules) if step_idx < len(s)]
        if not active_indices:
            break

        step_prompts = []
        for idx in active_indices:
            setting_name = schedules[idx][step_idx]
            forced_line_prefix = f"\n- Setting: {setting_name} | Location:"
            running_completions[idx] += forced_line_prefix
            step_prompts.append(base_chat_prompts[idx] + running_completions[idx])

        step_inputs = tokenizer(step_prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            step_outs = model.generate(
                **step_inputs,
                max_new_tokens=56,
                do_sample=True,
                temperature=args.temperature,
                min_p=args.min_p,
                top_p=args.top_p,
                eos_token_id=line_stop_ids,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        step_in_lens = [len(x) for x in step_inputs.input_ids]
        for local_i, global_idx in enumerate(active_indices):
            gen_line = tokenizer.decode(
                step_outs[local_i][step_in_lens[local_i]:], skip_special_tokens=True
            ).split("\n")[0].strip()
            running_completions[global_idx] += f" {gen_line}"

    return running_completions


def generate_batch_full_free(model, tokenizer, batch_prompts, args):
    """Single-pass unconstrained generation with calibrated sampling and full ChatML template."""
    chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in batch_prompts
    ]
    stop_ids = [tokenizer.eos_token_id]
    if "<end_of_turn>" in tokenizer.vocab:
        stop_ids.append(tokenizer.vocab["<end_of_turn>"])

    inputs = tokenizer(chat_prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1536,
            do_sample=True,
            temperature=args.temperature,
            min_p=args.min_p,
            top_p=args.top_p,
            eos_token_id=stop_ids,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

    in_lens = [len(x) for x in inputs.input_ids]
    completions = []
    for i, out_seq in enumerate(outputs):
        text = tokenizer.decode(out_seq[in_lens[i]:], skip_special_tokens=True).strip()
        completions.append(text)
    return completions


def run_generation(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    adapter_base = os.path.join(CHKPT_DIR, f"adapter_{args.split_tag}")
    ckpt_path = find_checkpoint_for_epoch(adapter_base, args.epoch)
    if not ckpt_path:
        raise FileNotFoundError(f"No checkpoint found for epoch={args.epoch} in {adapter_base}")

    model, tokenizer = load_model_and_tokenizer(args.base_model, ckpt_path)
    line_stop_ids = get_newline_token_ids(tokenizer)

    val_file = os.path.join(DATA_DIR, "val_causal.jsonl")
    df_val = pd.read_json(val_file, lines=True)
    if args.samples > 0:
        df_val = df_val.head(args.samples)

    prompts = df_val["prompt"].tolist()
    part_ids = df_val["part_id"].astype(str).tolist()

    print(
        f"\n🚀 Generating {len(df_val)} diaries | Mode: {args.mode} | "
        f"Epoch: {args.epoch} | Temp: {args.temperature} | min_p: {args.min_p}"
    )

    results = []
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating Batches"):
        b_prompts = prompts[i : i + args.batch_size]
        b_ids = part_ids[i : i + args.batch_size]

        if args.mode == "plan_guided":
            b_completions = generate_batch_plan_guided(
                model, tokenizer, b_prompts, args, line_stop_ids
            )
        else:
            b_completions = generate_batch_full_free(model, tokenizer, b_prompts, args)

        for pid, prm, comp in zip(b_ids, b_prompts, b_completions):
            results.append({
                "part_id": pid,
                "mode": args.mode,
                "prompt": prm,
                "generated_completion": comp,
            })

    out_tag = f"generated_{args.split_tag}_{args.mode}_epoch_{args.epoch}.jsonl"
    out_file = os.path.join(OUT_DIR, out_tag)
    pd.DataFrame(results).to_json(out_file, orient="records", lines=True)
    print(f"\n✅ Saved {len(results)} generated records -> {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--split_tag", type=str, default="100_percent")
    parser.add_argument("--epoch", type=str, default="6")
    parser.add_argument("--mode", type=str, choices=["plan_guided", "full_free"], default="plan_guided")
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--min_p", type=float, default=0.03)
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--max_contacts", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--samples", type=int, default=-1)
    args = parser.parse_args()

    run_generation(args)