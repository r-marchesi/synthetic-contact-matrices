import os
import sys
import json
import argparse
import torch
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
EXP04_DIR = os.path.abspath(os.path.join(EXP_DIR, "../04_causal_hierarchy"))
EXP04_SCRIPTS = os.path.join(EXP04_DIR, "scripts")
OUT_DATA_DIR = os.path.join(EXP_DIR, "data")

sys.path.append(EXP04_SCRIPTS)
from importlib import import_module
gen04 = import_module("03_generate")


def sample_training_candidates(args):
    os.makedirs(OUT_DATA_DIR, exist_ok=True)

    adapter_base = os.path.join(EXP04_DIR, "checkpoints", f"adapter_{args.split_tag}")
    ckpt_path = gen04.find_checkpoint_for_epoch(adapter_base, args.sft_epoch)
    if not ckpt_path:
        raise FileNotFoundError(f"Could not find SFT checkpoint for epoch {args.sft_epoch} in {adapter_base}")

    model, tokenizer = gen04.load_model_and_tokenizer(args.base_model, ckpt_path)
    line_stop_ids = gen04.get_newline_token_ids(tokenizer)

    train_file = os.path.join(EXP04_DIR, "data", f"train_causal_{args.split_tag}.jsonl")
    df_train = pd.read_json(train_file, lines=True)

    if args.max_participants > 0 and len(df_train) > args.max_participants:
        df_train = df_train.sample(n=args.max_participants, random_state=42).reset_index(drop=True)

    print(
        f"\n🚀 Sampling K={args.num_candidates} candidates for {len(df_train)} training participants "
        f"(Total generations: {len(df_train) * args.num_candidates})..."
    )

    # Expand each participant K times so we can batch across candidates directly
    expanded_rows = []
    for _, row in df_train.iterrows():
        for k in range(args.num_candidates):
            expanded_rows.append({
                "part_id": str(row["part_id"]),
                "cand_idx": k,
                "demos": row["demos"],
                "prompt": row["prompt"],
                "gt_completion": row["completion"],
            })

    gen_args = argparse.Namespace(
        temperature=args.temperature,
        min_p=args.min_p,
        top_p=args.top_p,
        max_contacts=args.max_contacts,
    )

    for i in tqdm(range(0, len(expanded_rows), args.batch_size), desc="Sampling Candidates"):
        batch = expanded_rows[i : i + args.batch_size]
        b_prompts = [r["prompt"] for r in batch]

        if args.mode == "plan_guided":
            b_Wait = gen04.generate_batch_plan_guided(model, tokenizer, b_prompts, gen_args, line_stop_ids)
        else:
            b_Wait = gen04.generate_batch_full_free(model, tokenizer, b_prompts, gen_args)

        for r, comp in zip(batch, b_Wait):
            r["candidate_completion"] = comp

    # Group candidates back by participant
    grouped = {}
    for r in expanded_rows:
        pid = r["part_id"]
        if pid not in grouped:
            grouped[pid] = {
                "part_id": pid,
                "demos": r["demos"],
                "prompt": r["prompt"],
                "gt_completion": r["gt_completion"],
                "candidates": [],
            }
        grouped[pid]["candidates"].append(r["candidate_completion"])

    out_path = os.path.join(OUT_DATA_DIR, f"candidates_{args.split_tag}.jsonl")
    with open(out_path, "w") as f:
        for rec in grouped.values():
            f.write(json.dumps(rec) + "\n")

    print(f"\n✅ Saved {len(grouped)} participant candidate pools -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--split_tag", type=str, default="100_percent")
    parser.add_argument("--sft_epoch", type=str, default="6")
    parser.add_argument("--mode", type=str, choices=["plan_guided", "full_free"], default="plan_guided")
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--max_participants", type=int, default=1500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min_p", type=float, default=0.02)
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--max_contacts", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    sample_training_candidates(args)