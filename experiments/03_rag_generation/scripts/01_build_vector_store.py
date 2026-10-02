import os
import re
import json
import faiss
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer

# --- PATH RESOLUTION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.abspath(os.path.join(EXP_DIR, "../02_structured_semantic/data"))
OUT_DIR = os.path.join(EXP_DIR, "data", "vector_store")

def extract_degree_from_macro(completion_text):
    """Parses the 'Total: X' value from the [MACRO] block."""
    match = re.search(r"Total:\s*(\d+)", completion_text)
    if match:
        return int(match.group(1))
    return 0

def build_vector_store(train_file):
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print(f"Loading training data from: {train_file}")
    records = []
    with open(train_file, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    print(f"Loaded {len(records)} training records. Parsing metadata...")
    
    metadata = []
    documents = [] # The text we actually embed (the DEMO prompt)
    
    for i, rec in enumerate(records):
        prompt = rec["prompt"].strip()
        completion = rec["completion"].strip()
        degree = extract_degree_from_macro(completion)
        
        # We embed the prompt because at inference, we only have the participant's prompt
        documents.append(prompt)
        
        metadata.append({
            "faiss_id": i,
            "part_id": rec.get("part_id", str(i)),
            "prompt": prompt,
            "completion": completion,
            "degree": degree
        })
        
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    # A fast, lightweight, locally-running model perfect for short semantic strings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Encoding documents into vectors...")
    embeddings = model.encode(documents, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    
    dimension = embeddings.shape[1]
    print(f"Initializing FAISS index with dimension {dimension}...")
    
    # Use inner product (cosine similarity) index since vectors will be normalized
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    # Save the index and metadata
    index_path = os.path.join(OUT_DIR, "train_demos.index")
    meta_path = os.path.join(OUT_DIR, "train_metadata.json")
    
    faiss.write_index(index, index_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"✅ FAISS Index saved to: {index_path}")
    print(f"✅ Metadata saved to: {meta_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file", 
        default=os.path.join(DATA_DIR, "train_dense.jsonl")
    )
    args = parser.parse_args()
    build_vector_store(args.train_file)