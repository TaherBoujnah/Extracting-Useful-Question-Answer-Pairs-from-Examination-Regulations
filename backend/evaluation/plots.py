from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _load(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid JSON: {path}")
    obj["_file"] = str(path)
    return obj


def _label(o: Dict[str, Any]) -> str:
    return (o.get("model_tag") or o.get("model") or Path(o["_file"]).stem).strip()


def _bar_plot(
    title: str,
    out_path: Path,
    labels: List[str],
    values: List[Optional[float]],
    ylabel: str,
    fmt: str = "{:.3f}",
):
    """
    Bar plot helper.

    Key behavior:
    - Missing values (None) are treated as missing (NaN), not as 0.
    - If *all* values are missing, emit a "No data (metric not computed)" figure
      instead of an empty 0-line chart.
    """
    fig = plt.figure(figsize=(11, 5.5))
    ax = plt.gca()

    # None -> NaN so matplotlib does not draw a misleading 0-height bar.
    plot_vals: List[float] = [float(v) if v is not None else float("nan") for v in values]
    missing = [math.isnan(v) for v in plot_vals]

    # If everything is missing, render an explicit message.
    if all(missing):
        ax.set_title(title)
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No data (metric not computed)",
            ha="center",
            va="center",
            fontsize=12,
        )
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    bars = ax.bar(range(len(labels)), plot_vals)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")

    # y-limit based only on finite values
    finite_vals = [v for v in plot_vals if not math.isnan(v)]
    vmax = max(finite_vals) if finite_vals else 1.0
    ymax = 1.0 if vmax == 0 else vmax * 1.15
    ax.set_ylim(bottom=0, top=ymax)

    for i, b in enumerate(bars):
        if missing[i]:
            txt = "NA"
            y = 0.0
        else:
            txt = fmt.format(float(plot_vals[i]))
            y = float(plot_vals[i])

        ax.text(
            b.get_x() + b.get_width() / 2,
            y,
            txt,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", type=str, required=True)
    ap.add_argument("--pattern", type=str, default="scores__*.json")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(eval_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No eval JSONs in {eval_dir} matching {args.pattern}")

    objs = [_load(p) for p in files]
    labels = [_label(o) for o in objs]

    metrics: List[Tuple[str, str, str, str]] = [
        ("Accuracy / Success rate", "accuracy_success_rate", "Rate (0..1)", "{:.3f}"),
        ("Coverage", "coverage", "Rate (0..1)", "{:.3f}"),
        ("Hallucination rate", "hallucination_rate", "Rate (0..1)", "{:.3f}"),
        ("Retrieval cosine (avg)", "avg_retrieval_cosine", "Cosine similarity", "{:.3f}"),
        ("Token F1 to retrieved chunk (avg)", "avg_token_f1", "Score (0..1)", "{:.3f}"),
        ("Roundtrip OK rate", "roundtrip_ok_rate", "Rate (0..1)", "{:.3f}"),
        ("Roundtrip F1 (avg)", "avg_roundtrip_f1", "Score (0..1)", "{:.3f}"),
        ("Runtime: generation", "generation_runtime_seconds", "Seconds", "{:.1f}"),
    ]

    # Save a simple table for debugging / writeup
    summary = []
    for o in objs:
        row = {"label": _label(o), "file": o["_file"]}
        for _, key, _, _ in metrics:
            row[key] = o.get(key)
        summary.append(row)

    (out_dir / "summary_table.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for title, key, ylabel, fmt in metrics:
        vals = [_safe_float(o.get(key)) for o in objs]
        _bar_plot(title, out_dir / f"{key}.png", labels, vals, ylabel, fmt=fmt)

    print(f"✅ Plots saved to: {out_dir}")
    print(f"✅ Summary table: {out_dir / 'summary_table.json'}")


if __name__ == "__main__":
    main()



