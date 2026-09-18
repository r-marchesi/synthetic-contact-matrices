import os
import glob
import json
import re
import argparse
import pandas as pd

# The exact reverse mapping to standardize column names for the evaluation script
NATURAL_KEYS = [
    "Contact Age", "Contact Gender", "Contact Frequency", 
    "Physical Contact", "Distance during Contact", 
    "Relationship to Participant", "Contact Setting", "Location of Contact"
]

def extract_json(text):
    """Parses standard JSON."""
    try:
        # Find the first [ and last ] to handle any stray generation tokens
        start = text.find('[')
        end = text.rfind(']') + 1
        if start != -1 and end != 0:
            return json.loads(text[start:end])
        return []
    except json.JSONDecodeError:
        return []

def extract_cot(text):
    """Chain of Thought ends with a JSON array. Extract it."""
    # The JSON array always starts after the reasoning block
    return extract_json(text)

def extract_yaml(text):
    """Parses the lightweight YAML format back into a list of dicts."""
    contacts = []
    current_contact = {}
    
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('- contact:'):
            if current_contact:
                contacts.append(current_contact)
            current_contact = {}
        elif ':' in line and current_contact is not None:
            parts = line.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                val = parts[1].strip()
                current_contact[key] = val
                
    if current_contact:
        contacts.append(current_contact)
    return contacts

def extract_nl(text):
    """Uses Regex to extract variables from the strict Natural Language template."""
    contacts = []
    # Regex template matching utils_prompting.py exactly
    pattern = re.compile(
        r"Contact \d+ was a (.*?)-year-old (.*?). "
        r"They met at (.*?) \((.*?)\). "
        r"They maintained a (.*?) distance. "
        r"The relationship is (.*?) and they meet (.*?). "
        r"Physical contact: (.*?)\."
    )
    
    matches = pattern.findall(text)
    for match in matches:
        contacts.append({
            "Contact Age": match[0],
            "Contact Gender": match[1],
            "Contact Setting": match[2],
            "Location of Contact": match[3],
            "Distance during Contact": match[4],
            "Relationship to Participant": match[5],
            "Contact Frequency": match[6],
            "Physical Contact": match[7]
        })
    return contacts

def parse_results(results_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    jsonl_files = glob.glob(os.path.join(results_dir, "*.jsonl"))
    
    if not jsonl_files:
        print(f"No .jsonl files found in {results_dir}")
        return

    for file_path in jsonl_files:
        filename = os.path.basename(file_path)
        csv_filename = filename.replace('.jsonl', '.csv')
        output_path = os.path.join(output_dir, csv_filename)
        
        print(f"Parsing: {filename}...")
        
        parsed_rows = []
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                part_id = data.get("part_id")
                prompt_style = data.get("prompt_style")
                raw_text = data.get("raw_generated_text", "")
                
                # Route to the correct parser
                if prompt_style == "json_strict":
                    contacts = extract_json(raw_text)
                elif prompt_style == "chain_of_thought":
                    contacts = extract_cot(raw_text)
                elif prompt_style == "yaml_style":
                    contacts = extract_yaml(raw_text)
                elif prompt_style == "natural_language":
                    contacts = extract_nl(raw_text)
                else:
                    contacts = []

                # Flatten into CSV rows
                for c in contacts:
                    row = {"part_id": part_id}
                    for key in NATURAL_KEYS:
                        row[key] = c.get(key, "Unknown")
                    parsed_rows.append(row)
                    
        # Save to CSV
        df = pd.DataFrame(parsed_rows)
        # Reorder columns for consistency
        cols = ["part_id"] + NATURAL_KEYS
        df = df.reindex(columns=cols)
        df.to_csv(output_path, index=False)
        print(f"--> Saved {len(df)} contacts to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="data/results")
    parser.add_argument("--output_dir", type=str, default="data/parsed_results")
    args = parser.parse_args()
    parse_results(args.results_dir, args.output_dir)