import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
EXP04_DIR = os.path.abspath(os.path.join(EXP_DIR, "../04_causal_hierarchy"))
EXP04_SCRIPTS = os.path.join(EXP04_DIR, "scripts")

GEN_DIR = os.path.join(EXP_DIR, "generated")
OUT_DIR = os.path.join(EXP_DIR, "results")

sys.path.append(EXP04_SCRIPTS)
from importlib import import_module
eval04 = import_module("04_evaluate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val_file",
        type=str,
        default=os.path.join(EXP04_DIR, "data", "val_causal.jsonl"),
    )
    args = parser.parse_args()

    eval04.GEN_DIR = GEN_DIR
    eval04.OUT_DIR = OUT_DIR
    eval04.run_evaluation(args)