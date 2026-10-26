import json
import os
import argparse

def check_missing_participants(jsonl_file):
    if not os.path.exists(jsonl_file):
        print(f"Error: Could not find {jsonl_file}")
        return

    valid_participants = 0
    true_zeros = 0
    parse_failures = []

    print(f"Scanning raw generations in: {jsonl_file}\n")

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            part_id = data.get("part_id")
            raw_text = data.get("raw_generated_text", "")

            # 1. Attempt to isolate the JSON array exactly as 005_parse_results.py does
            start = raw_text.find('[')
            end = raw_text.rfind(']') + 1
            
            if start != -1 and end != 0:
                json_string = raw_text[start:end]
                try:
                    # 2. Try to parse it
                    contacts = json.loads(json_string)
                    
                    if isinstance(contacts, list):
                        if len(contacts) == 0:
                            true_zeros += 1  # Model explicitly output an empty array
                        else:
                            valid_participants += 1  # Standard valid generation
                    else:
                        parse_failures.append((part_id, "Parsed successfully, but result was not a list.", raw_text))
                        
                except json.JSONDecodeError as e:
                    # 3. Model generated brackets, but the inside is broken JSON
                    parse_failures.append((part_id, f"JSON Decode Error: {e}", raw_text))
            else:
                # 4. Model failed to generate the array brackets at all
                parse_failures.append((part_id, "Missing JSON array brackets '[' and ']'.", raw_text))

    # --- Print Diagnostics ---
    total = valid_participants + true_zeros + len(parse_failures)
    print(f"Total Participants Analyzed: {total}")
    print("-" * 40)
    print(f"✅ Valid Contacts Generated:  {valid_participants}")
    print(f"🎯 True Zero Contacts:        {true_zeros} (Model successfully generated '[]')")
    print(f"❌ Parse Failures:            {len(parse_failures)} (Structure was malformed)")
    
    if parse_failures:
        print("\n==================================================")
        print("SAMPLE PARSE FAILURES")
        print("==================================================")
        # Show the first 3 failures so you can see exactly how the LLM broke the formatting
        for pid, reason, text in parse_failures[:3]:
            print(f"\nParticipant ID: {pid}")
            print(f"Failure Reason: {reason}")
            print(f"Raw Output Snippet:\n{text[:300]}... [TRUNCATED]" if len(text) > 300 else f"Raw Output:\n{text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Defaulting to the file you've been working with
    parser.add_argument("--file", type=str, default="data/results/prompt_cot_epoch_2_results.jsonl")
    args = parser.parse_args()
    
    check_missing_participants(args.file)