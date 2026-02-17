# backend/evaluation/run_pilot.py
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str]):
    print("\n>>> " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roundtrip_model", type=str, default=None)
    ap.add_argument("--entailment_model", type=str, default=None)
    args = ap.parse_args()

    files = [
        ("data/generated/faqs_hybrid__gemma3_4b.jsonl",  "hybrid gemma3_4b"),
        ("data/generated/faqs_hybrid__llama3.1_8b.jsonl","hybrid llama3.1_8b"),
        ("data/generated/faqs_hybrid__qwen2.5_7b.jsonl", "hybrid qwen2.5_7b"),
        ("data/generated/faqs_slow_hq__llama3.1_8b.jsonl","slow_hq llama3.1_8b"),
        ("data/generated/faqs_slow_hq__qwen2.5_7b.jsonl", "slow_hq qwen2.5_7b"),
    ]

    for inp, tag in files:
        out = f"data/eval/scores__{tag.replace(' ', '__')}.json"
        cmd = [
            "python", "-m", "backend.evaluation.layer1_jsonl",
            "--input", inp,
            "--output", out,
            "--model_tag", tag,
        ]
        if args.roundtrip_model:
            cmd += ["--roundtrip_model", args.roundtrip_model]
        if args.entailment_model:
            cmd += ["--entailment_model", args.entailment_model]
        run(cmd)


if __name__ == "__main__":
    main()

