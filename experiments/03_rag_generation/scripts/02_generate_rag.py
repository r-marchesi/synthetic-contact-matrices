import os
import json
import faiss
import torch
import argparse
import numpy as np
from tqdm import tqdm
import transformers
from sentence_transformers import SentenceTransformer

import guidance
from guidance import models, select, gen, capture

# --- PATH RESOLUTION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
VECTOR_DIR = os.path.join(EXP_DIR, "data", "vector_store")
VAL_DATA = os.path.abspath(os.path.join(EXP_DIR, "../02_structured_semantic/data/val_dense.jsonl"))
OUT_DIR = os.path.join(EXP_DIR, "generated")

def load_empirical_rules():
    grammar_path = os.path.join(VECTOR_DIR, "empirical_grammar.json")
    
    if os.path.exists(grammar_path):
        print(f"Loading empirical data-driven bounds from {grammar_path}")
        with open(grammar_path, "r") as f:
            rules = json.load(f)
        settings = rules.get("valid_settings", ["Home", "Work", "School", "Other"])
        freqs = rules.get("valid_frequencies", ["Daily", "Weekly", "Monthly", "Unknown"])
        dists = rules.get("valid_distances", ["Physical", "Non-physical", "Unknown"])
        max_age_digits = rules.get("max_age_digits", 3)
    else:
        print("empirical_grammar.json not found. Using default bounds.")
        settings = ["Home", "Work", "School", "Other"]
        freqs = ["Daily", "Weekly", "Monthly", "Unknown"]
        dists = ["Physical", "Non-physical", "Unknown"]
        max_age_digits = 3
        
    return settings, freqs, dists, max_age_digits

def select_variance_anchors(pool_metadata):
    if not pool_metadata: return []
    
    sorted_pool = sorted(pool_metadata, key=lambda x: x["degree"])
    isolate = sorted_pool[0]
    super_spreader = sorted_pool[-1]
    median = sorted_pool[len(sorted_pool) // 2]
    busy = sorted_pool[int(len(sorted_pool) * 0.75)]
    
    anchors = []
    seen = set()
    for cand in [median, super_spreader, isolate, busy]:
        if cand["part_id"] not in seen:
            anchors.append(cand)
            seen.add(cand["part_id"])
            
    return anchors

def build_rag_prompt(anchors, target_prompt):
    sys_instruction = (
        "You are an epidemiological data synthesizer. Your task is to generate a realistic "
        "daily contact diary for a target participant.\n"
        "I will provide you with examples of real diaries from similar people. Notice that "
        "real human behavior is highly variable: some have very few contacts, while others "
        "have many. Do not copy these examples exactly. Use them to understand the *range* "
        "of possible behaviors, and generate a new, unique diary.\n\n"
    )
    
    examples_str = "[REAL EXAMPLES FOR CONTEXT]\n"
    for i, anchor in enumerate(anchors):
        examples_str += f"Example {i+1}:\n{anchor['prompt']}\n{anchor['completion']}\n\n"
        
    target_str = f"[TARGET PARTICIPANT - GENERATE DIARY]\n{target_prompt}\n"
    
    return sys_instruction + examples_str + target_str

@guidance
def generate_constrained_diary(lm, settings, freqs, dists, max_age_digits, max_contacts=25):
    lm += "[MACRO] Total: " + capture(gen(regex=r"\d{1,2}"), name="total") + " | "
    lm += "Home: " + gen(regex=r"\d{1,2}") + " | "
    lm += "Work: " + gen(regex=r"\d{1,2}") + " | "
    lm += "School: " + gen(regex=r"\d{1,2}") + " | "
    lm += "Other: " + gen(regex=r"\d{1,2}") + "\n"
    
    lm += "[CONTACTS]\n"
    
    total_contacts = int(lm["total"])
    
    if total_contacts == 0:
        lm += "- None\n"
    else:
        for i in range(min(total_contacts, max_contacts)):
            lm += "- Setting: " + select(settings) + " | "
            # Uses empirical age digits limit extracted from your dataset
            lm += "Age: " + gen(regex=rf"(\d{{1,{max_age_digits}}}|Unknown)") + " | "
            lm += "Gender: " + select(["M", "F", "Unknown"]) + " | "
            lm += "Freq: " + select(freqs) + " | "
            lm += "Dist: " + select(dists) + "\n"
            
    return lm

def generate_rag_data(model_name="google/gemma-2-9b-it", num_samples=100):
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("Loading embedding model and FAISS index...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index(os.path.join(VECTOR_DIR, "train_demos.index"))
    
    with open(os.path.join(VECTOR_DIR, "train_metadata.json"), "r") as f:
        metadata = json.load(f)
        
    print(f"Loading Guidance Model ({model_name})...")
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    lm = models.Transformers(hf_model, hf_tokenizer)
    
    # Load rules for the constraint engine
    settings, freqs, dists, max_age_digits = load_empirical_rules()
    
    print(f"Loading validation targets from {VAL_DATA}...")
    val_records = []
    with open(VAL_DATA, "r") as f:
        for line in f:
            if line.strip():
                val_records.append(json.loads(line))
                
    val_records = val_records[:num_samples]
    results = []
    
    print("Starting Constrained RAG Generation...")
    for rec in tqdm(val_records):
        target_prompt = rec["prompt"].strip()
        
        query_vector = embedder.encode([target_prompt]).astype("float32")
        faiss.normalize_L2(query_vector)
        distances, indices = index.search(query_vector, k=50)
        pool = [metadata[idx] for idx in indices[0] if idx != -1]
        anchors = select_variance_anchors(pool)
        
        full_prompt = build_rag_prompt(anchors, target_prompt)
        
        state = lm + full_prompt
        state = state + generate_constrained_diary(settings, freqs, dists, max_age_digits)
        
        generated_completion = str(state)[len(full_prompt):]
        
        results.append({
            "part_id": rec.get("part_id"),
            "prompt": target_prompt,
            "true_completion": rec["completion"],
            "rag_completion": generated_completion.strip()
        })
        
    out_file = os.path.join(OUT_DIR, "rag_generated_contacts.jsonl")
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    print(f"\n✅ Generated {len(results)} structured RAG diaries -> {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    
    generate_rag_data(num_samples=args.num_samples)