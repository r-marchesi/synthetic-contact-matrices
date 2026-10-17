import os
import sys
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
EXP04_DIR = os.path.abspath(os.path.join(EXP_DIR, "../04_causal_hierarchy"))
EXP04_SCRIPTS = os.path.join(EXP04_DIR, "scripts")

INDEX_DIR = os.path.join(EXP_DIR, "data")
OUT_DIR = os.path.join(EXP_DIR, "generated")

sys.path.append(EXP04_SCRIPTS)
from importlib import import_module
gen04 = import_module("03_generate")


def safe_float(val, default=-1.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def age_life_stage(age_val):
    if age_val < 0:
        return "unknown"
    if age_val < 6:
        return "infant_toddler"
    if age_val < 19:
        return "school_age"
    if age_val < 30:
        return "young_adult"
    if age_val < 65:
        return "working_adult"
    return "senior"


def compute_gower_distances(target_demos, index_records):
    """
    Computes an epidemiologically weighted Gower distance between target_demos
    and all training profiles in index_records. Lower distance = closer match.
    """
    t_age = safe_float(target_demos.get("age"), -1.0)
    t_stage = age_life_stage(t_age)
    t_hh = safe_float(target_demos.get("hh_size"), 1.0)
    t_work = str(target_demos.get("work_today", "")).lower()
    t_school = str(target_demos.get("school_today", "")).lower()
    t_occ = str(target_demos.get("occupation", "")).lower()
    t_gen = str(target_demos.get("gender", "")).lower()
    t_edu = str(target_demos.get("education", "")).lower()
    t_inc = str(target_demos.get("income", "")).lower()
    t_reg = str(target_demos.get("region", "")).lower()

    distances = np.zeros(len(index_records), dtype=np.float32)

    for i, rec in enumerate(index_records):
        d_demos = rec["demos"]
        dist = 0.0

        # 1. Age distance (weight 3.0) + Life-Stage Cohort match (weight 2.0)
        c_age = rec["age_num"]
        if t_age >= 0 and c_age >= 0:
            dist += 3.0 * min(1.0, abs(t_age - c_age) / 50.0)
        else:
            dist += 1.5
        if t_stage != rec["life_stage"]:
            dist += 2.0

        # 2. Daily Work & School Attendance (weight 2.5 each - critical drivers of volume)
        if t_work != str(d_demos.get("work_today", "")).lower():
            dist += 2.5
        if t_school != str(d_demos.get("school_today", "")).lower():
            dist += 2.5

        # 3. Household Size distance (weight 2.0)
        c_hh = rec["hh_size_num"]
        dist += 2.0 * min(1.0, abs(t_hh - c_hh) / 5.0)

        # 4. Occupation, Gender, Education, Income, Region
        if t_occ != str(d_demos.get("occupation", "")).lower():
            dist += 1.2
        if t_gen != str(d_demos.get("gender", "")).lower():
            dist += 0.6
        if t_edu != str(d_demos.get("education", "")).lower():
            dist += 0.5
        if t_inc != str(d_demos.get("income", "")).lower():
            dist += 0.4
        if t_reg != str(d_demos.get("region", "")).lower():
            dist += 0.3

        distances[i] = dist

    return distances


def select_stratified_quantile_anchors(neighbor_pool, rng):
    """
    Sorts the K nearest demographic neighbors by contact degree and selects 4 anchors
    from jittered empirical quantiles so the prompt represents the true conditional
    distribution without 100th-percentile super-spreader distortion.
    """
    if not neighbor_pool:
        return []
    sorted_pool = sorted(neighbor_pool, key=lambda x: x["degree"])
    n = len(sorted_pool)
    if n <= 4:
        return sorted_pool

    quantile_bands = [
        (0.05, 0.25),
        (0.28, 0.48),
        (0.52, 0.72),
        (0.75, 0.94),
    ]
    anchors = []
    used_ids = set()

    for q_low, q_high in quantile_bands:
        q = rng.uniform(q_low, q_high)
        idx = min(n - 1, max(0, int(round(q * (n - 1)))))
        cand = sorted_pool[idx]
        if cand["part_id"] not in used_ids:
            anchors.append(cand)
            used_ids.add(cand["part_id"])

    # Shuffle anchor presentation order so position bias doesn't favor the last quantile
    rng.shuffle(anchors)
    return anchors


def build_rag_prompt(anchors, target_prompt):
    """
    Wraps the 4 retrieved demographic neighbor diaries as in-context examples,
    ending with the exact target participant prompt.
    """
    header = (
        "You are an epidemiological synthetic data generator. Below are 4 real contact diaries "
        "from demographic peers who share similar age, household size, and work/school attendance.\n"
        "Observe how [PLAN] counts match the [CONTACTS] list and how contact roles and ages "
        "correlate with the participant's demographics.\n\n"
    )
    ex_blocks = []
    for i, anc in enumerate(anchors, 1):
        p_block = anc["prompt"].replace(
            "Generate the daily epidemiological contact diary for the following participant:\n", ""
        )
        ex_blocks.append(f"--- Peer Example {i} ---\n{p_block}\n{anc['completion']}")

    examples_section = "\n\n".join(ex_blocks)
    return (
        f"{header}{examples_section}\n\n"
        f"--- Target Participant ---\n"
        f"{target_prompt}"
    )


def load_rag_model(base_model_id, use_sft_adapter, split_tag, sft_epoch):
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading base model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    if use_sft_adapter:
        adapter_base = os.path.join(EXP04_DIR, "checkpoints", f"adapter_{split_tag}")
        ckpt_path = gen04.find_checkpoint_for_epoch(adapter_base, sft_epoch)
        if not ckpt_path:
            raise FileNotFoundError(f"SFT checkpoint not found in {adapter_base}")
        print(f"Merging SFT LoRA adapter for Hybrid RAG+SFT: {ckpt_path}")
        model = PeftModel.from_pretrained(model, ckpt_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def run_tabular_rag_generation(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(args.seed)

    index_path = os.path.join(INDEX_DIR, f"tabular_index_{args.split_tag}.json")
    with open(index_path, "r") as f:
        index_records = json.load(f)

    val_file = os.path.join(EXP04_DIR, "data", "val_causal.jsonl")
    df_val = pd.read_json(val_file, lines=True)
    if args.samples > 0:
        df_val = df_val.head(args.samples)

    print(f"Retrieving k={args.top_k} Gower demographic neighbors for {len(df_val)} validation targets...")
    rag_prompts = []
    part_ids = []

    for _, row in df_val.iterrows():
        pid = str(row["part_id"]).strip()
        t_demos = row["demos"]
        t_prompt = row["prompt"]

        dists = compute_gower_distances(t_demos, index_records)
        nearest_idx = np.argsort(dists)[: args.top_k]
        pool = [index_records[idx] for idx in nearest_idx]
        anchors = select_stratified_quantile_anchors(pool, rng)

        rag_prompts.append(build_rag_prompt(anchors, t_prompt))
        part_ids.append(pid)

    model, tokenizer = load_rag_model(
        args.base_model, args.use_sft_adapter, args.split_tag, args.sft_epoch
    )
    line_stop_ids = gen04.get_newline_token_ids(tokenizer)

    tag_prefix = f"hybrid_sft_ep{args.sft_epoch}" if args.use_sft_adapter else "zeroshot_base"
    print(f"\n🚀 Running Batched Plan-Guided RAG Generation ({tag_prefix})...")

    results = []
    for i in tqdm(range(0, len(rag_prompts), args.batch_size), desc="RAG Batches"):
        b_prompts = rag_prompts[i : i + args.batch_size]
        b_ids = part_ids[i : i + args.batch_size]

        b_completions = gen04.generate_batch_plan_guided(
            model, tokenizer, b_prompts, args, line_stop_ids
        )

        for pid, comp in zip(b_ids, b_completions):
            results.append({
                "part_id": pid,
                "mode": f"rag_{tag_prefix}",
                "generated_completion": comp,
            })

    out_file = os.path.join(OUT_DIR, f"generated_rag_{tag_prefix}_{args.split_tag}.jsonl")
    pd.DataFrame(results).to_json(out_file, orient="records", lines=True)
    print(f"\n✅ Saved {len(results)} Tabular-RAG diaries -> {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--split_tag", type=str, default="100_percent")
    parser.add_argument("--use_sft_adapter", action="store_true")
    parser.add_argument("--sft_epoch", type=str, default="6")
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.90)
    parser.add_argument("--min_p", type=float, default=0.03)
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--max_contacts", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_tabular_rag_generation(args)