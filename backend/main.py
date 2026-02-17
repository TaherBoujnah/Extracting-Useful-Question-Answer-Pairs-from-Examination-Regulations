# backend/main.py
import time
import argparse
from pathlib import Path

from backend.pdf_parser import run_parser as run_pdf_parser
from backend.chunker import run_chunker as run_chunker
from backend.span_extractor import run_span_extractor
from backend.generators.llama_31 import generate_qas as generate_llama31
from backend.evaluation.layer1 import evaluate_file
from backend.evaluation.plots import plot_comparison


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def timed(label, fn, *args, **kwargs):
    t0 = time.perf_counter()
    print(f"\n▶ {label}")
    result = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    print(f"⏱ {label} finished in {dt:.2f}s")
    return result, dt


def run_pipeline(limit_spans=None, limit_sections=None, eval_only=False, gen_only=False, skip_plots=False):
    runtimes = {}

    # ---------- Stage 1: PDF → raw_pages.jsonl ----------
    if not eval_only and not gen_only:
        _, dt = timed("PDF parsing (PDF → raw_pages.jsonl)", run_pdf_parser)
        runtimes["pdf_parser"] = dt

        # ---------- Stage 2: raw → exam_regulations.txt ----------
        _, dt = timed("Text build (raw_pages.jsonl → exam_regulations.txt)", run_chunker)
        runtimes["chunker"] = dt

        

    # ---------- Stage 4: generation ----------
    generated_path = DATA_DIR / "generated_llama31.json"
    if not eval_only:
        _, dt = timed(
            "FAQ generation (spans.json → generated_llama31.json) [Llama 3.1 8B via Ollama]",
            generate_llama31,
            limit=limit_spans,
            checkpoint_every=25
        )
        runtimes["generation_llama31"] = dt

    # ---------- Stage 5: evaluation layer 1 ----------
    scores_path = DATA_DIR / "scores_llama31.json"
    _, dt = timed(
        "Layer 1 evaluation (generated → scores)",
        evaluate_file,
        input_path=generated_path,
        output_path=scores_path
    )
    runtimes["evaluation_layer1"] = dt

    # ---------- Stage 6: plots ----------
    if not skip_plots:
        # plot_comparison expects list of score files
        _, dt = timed(
            "Plotting (scores → PNG charts)",
            plot_comparison,
            [str(scores_path)],
            DATA_DIR / "plots"
        )
        runtimes["plots"] = dt

    # ---------- Save runtime report ----------
    runtime_report = DATA_DIR / "runtime_report.json"
    runtime_report.write_text(
        __import__("json").dumps(runtimes, indent=2),
        encoding="utf-8"
    )
    print(f"\n✅ Pipeline complete. Runtime report saved to: {runtime_report}")
    print(f"📁 Outputs: {DATA_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit_spans", type=int, default=None, help="Generate only from first N spans (debug).")
    parser.add_argument("--limit_sections", type=int, default=None, help="Extract spans from first N sections (debug).")
    parser.add_argument("--eval_only", action="store_true", help="Skip parsing/extraction/generation, only run eval.")
    parser.add_argument("--gen_only", action="store_true", help="Skip parsing/extraction, only run generation+eval.")
    parser.add_argument("--skip_plots", action="store_true", help="Do not generate PNG plots.")
    args = parser.parse_args()

    run_pipeline(
        limit_spans=args.limit_spans,
        limit_sections=args.limit_sections,
        eval_only=args.eval_only,
        gen_only=args.gen_only,
        skip_plots=args.skip_plots
    )




