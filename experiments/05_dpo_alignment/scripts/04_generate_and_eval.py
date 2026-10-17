import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
EXP04_DIR = os.path.abspath(os.path.join(EXP_DIR, "../04_causal_hierarchy"))
EXP04_SCRIPTS = os.path.join(EXP04_DIR, "scripts")

CHKPT_DIR = os.path.join(EXP_DIR, "checkpoints")
GEN_DIR = os.path.join(EXP_DIR, "generated")
OUT_DIR = os.path.join(EXP_DIR, "results")

sys.path.append(EXP04_SCRIPTS)
from importlib import import_module
gen04 = import_module("03_generate")
eval04 = import_module("04_evaluate")


def generate_and_evaluate_dpo(args):
    os.makedirs(GEN_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    adapter_base = os.path.join(CHKPT_DIR, f"dpo_adapter_{args.split_tag}")
    ckpt_path = gen04.find_checkpoint_for_epoch(adapter_base, args.dpo_epoch)
    if not ckpt_path:
        raise FileNotFoundError(f"Could not find DPO checkpoint in {adapter_base}")

    model, tokenizer = gen04.load_model_and_tokenizer(args.base_model, ckpt_path)
    line_stop_ids = gen04.get_newline_token_ids(tokenizer)

    val_file = os.path.join(EXP04_DIR, "data", "val_causal.jsonl")
    df_val = pd.read_json(val_file, lines=True)
    if args.samples > 0:
        df_val = df_val.head(args.samples)

    prompts = df_val["prompt"].tolist()
    part_ids = df_val["part_id"].astype(str).tolist()

    results = []
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating DPO Validation Diaries"):
        b_prompts = prompts[i : i + args.batch_size]
        b_ids = part_ids[i : i + args.batch_size]

        if args.mode == "plan_guided":
            b_comp = gen04.generate_batch_plan_guided(model, tokenizer, b_prompts, args, line_stop_ids)
        else:
            b_comp = gen04.generate_batch_full_free(model, tokenizer, b_prompts, args)

        for pid, prm, comp in zip(b_ids, b_prompts, b_comp):
            results.append({
                "part_id": pid,
                "mode": f"dpo_{args.mode}",
                "prompt": prm,
                "generated_completion": comp,
            })

    out_tag = f"generated_dpo_{args.split_tag}_{args.mode}_epoch_{args.dpo_epoch}.jsonl"
    out_file = os.path.join(GEN_DIR, out_tag)
    pd.DataFrame(results).to_json(out_file, orient="records", lines=True)
    print(f"✅ Saved DPO validation generations -> {out_file}")

    # Point eval04 directories to Exp 05 and run the full evaluation suite
    eval04.GEN_DIR = GEN_DIR
    eval04.OUT_DIR = OUT_DIR
    eval_args = argparse.Namespace(val_file=val_file)
    eval04.run_evaluation(eval_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--split_tag", type=str, default="100_percent")
    parser.add_argument("--dpo_epoch", type=str, default="final")
    parser.add_argument("--mode", type=str, choices=["plan_guided", "full_free"], default="plan_guided")
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--min_p", type=float, default=0.03)
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--max_contacts", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--samples", type=int, default=-1)
    args = parser.parse_args()

    generate_and_evaluate_dpo(args)