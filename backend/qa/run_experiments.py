import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def run(cmd: list[str]):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_limit", type=int, default=10)  # pilot by default
    ap.add_argument("--include_tiny", action="store_true")
    args = ap.parse_args()

    models = ["qwen2.5:7b", "llama3.1:7b"]
    if args.include_tiny:
        models.append("qwen2.5:0.5b")

    # Slow/HQ only for 7B models (tiny would be pointless here)
    for m in ["qwen2.5:7b", "llama3.1:7b"]:
        run(["python", "-m", "backend.qa.generate_slow_hq", "--model", m, "--seed_limit", str(args.seed_limit)])

    # Hybrid for all chosen models
    for m in models:
        run(["python", "-m", "backend.qa.generate_hybrid", "--model", m, "--seed_limit", str(args.seed_limit)])
