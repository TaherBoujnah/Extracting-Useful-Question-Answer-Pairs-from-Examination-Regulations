# backend/qa/kmeans_select_and_lda_plot.py
# Select a cohesive 50–100 FAQ cluster in embedding space, then plot with LDA so the chosen cluster is visibly separated.
# Run on your DEDUP file (recommended).

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def quality_score(qa: Dict) -> float:
    q = (qa.get("question") or "").strip()
    a = (qa.get("answer") or "").strip()
    s = 0.0
    s += min(len(a), 500) / 500.0
    s += 0.2 if q.endswith("?") else 0.0
    s += 0.2 if ("§" in q or "§" in a) else 0.0
    s += 0.1 if any(ch.isdigit() for ch in a) else 0.0
    s -= 0.3 if len(a) < 40 else 0.0
    return float(s)


def cosine_dist_to_centroid(emb: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    # emb is normalized
    X = emb[idxs]
    centroid = X.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
    sims = X @ centroid
    return 1.0 - sims  # cosine distance


def choose_most_cohesive_cluster(labels: np.ndarray, emb: np.ndarray, target_min: int, target_max: int) -> Tuple[int, np.ndarray, List[Tuple[int, int, float]]]:
    stats: List[Tuple[int, int, float]] = []
    for lab in np.unique(labels):
        idxs = np.where(labels == lab)[0]
        d = cosine_dist_to_centroid(emb, idxs)
        cohesion = float(d.mean())  # lower = tighter
        stats.append((int(lab), int(len(idxs)), cohesion))

    candidates = [s for s in stats if s[1] >= target_min]
    if candidates:
        # prefer tightness, then closeness to target_max
        candidates.sort(key=lambda x: (x[2], abs(x[1] - target_max)))
        chosen = candidates[0][0]
    else:
        # fallback: largest cluster if nothing meets min size
        stats.sort(key=lambda x: x[1], reverse=True)
        chosen = stats[0][0]

    chosen_idxs = np.where(labels == chosen)[0]
    return int(chosen), chosen_idxs, stats


def add_cov_ellipse(ax, pts: np.ndarray, percentile: float = 95.0):
    """
    Draw an ellipse capturing ~percentile% of points assuming roughly elliptical spread.
    This is for visualization only.
    """
    if pts.shape[0] < 5:
        return

    mean = pts.mean(axis=0)
    cov = np.cov(pts.T)

    # eigen-decomposition
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    # angle in degrees
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    # scale factor: use chi-square quantile for 2D.
    # Common approximations:
    # 90% -> 4.605, 95% -> 5.991, 99% -> 9.210
    chi2 = {90.0: 4.605, 95.0: 5.991, 99.0: 9.210}.get(percentile, 5.991)

    width = 2.0 * np.sqrt(vals[0] * chi2)
    height = 2.0 * np.sqrt(vals[1] * chi2)

    e = Ellipse(xy=mean, width=width, height=height, angle=angle, fill=False, linewidth=2)
    ax.add_patch(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True, help="Prefer deduplicated JSONL")
    ap.add_argument("--out_jsonl", required=True, help="Final selected FAQs JSONL")
    ap.add_argument("--plot_png", required=True, help="LDA scatter plot")
    ap.add_argument("--embed_model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    ap.add_argument("--n_clusters", type=int, default=12)
    ap.add_argument("--target_min", type=int, default=50)
    ap.add_argument("--target_max", type=int, default=100)

    ap.add_argument("--ellipse_percentile", type=float, default=95.0, help="90, 95, or 99 are good")
    ap.add_argument("--show_other_clusters", action="store_true", help="If set, color each cluster differently (busy).")

    args = ap.parse_args()

    rows = read_jsonl(Path(args.input_jsonl))
    rows = [r for r in rows if (r.get("question") or "").strip() and (r.get("answer") or "").strip()]
    if not rows:
        raise SystemExit("No valid rows found in input.")

    questions = [r["question"].strip() for r in rows]
    scores = np.array([quality_score(r) for r in rows], dtype=np.float32)

    print(f"Loaded {len(rows)} FAQs")
    print("Embedding questions...")
    model = SentenceTransformer(args.embed_model)
    emb = model.encode(questions, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)

    print(f"Running KMeans (K={args.n_clusters})...")
    km = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(emb)

    chosen_label, chosen_idxs, stats = choose_most_cohesive_cluster(labels, emb, args.target_min, args.target_max)

    # Trim to target_max by centrality (closest to centroid)
    dists = cosine_dist_to_centroid(emb, chosen_idxs)
    chosen_idxs = chosen_idxs[np.argsort(dists)]
    if len(chosen_idxs) > args.target_max:
        chosen_idxs = chosen_idxs[: args.target_max]

    # (Optional) if you want to prefer "quality" among equally central points, you can combine score + distance.
    selected = [rows[i] for i in chosen_idxs]
    write_jsonl(Path(args.out_jsonl), selected)

    # LDA projection (THIS is what makes clusters visually separable)
    # Needs labels; n_components must be <= (n_classes-1) so with K>=3 we can do 2D.
    print("Projecting to 2D with LDA (for visible separation)...")
    lda = LDA(n_components=2)
    XY = lda.fit_transform(emb, labels)

    # Plot
    Path(args.plot_png).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))

    if args.show_other_clusters:
        # Each cluster in a different color (can be visually busy)
        for lab in np.unique(labels):
            idxs = np.where(labels == lab)[0]
            ax.scatter(XY[idxs, 0], XY[idxs, 1], s=10, alpha=0.25)
    else:
        # All non-selected in faint gray
        mask = np.ones(len(rows), dtype=bool)
        mask[chosen_idxs] = False
        ax.scatter(XY[mask, 0], XY[mask, 1], s=10, alpha=0.12, label="Other questions")

    # Selected cluster highlighted
    ax.scatter(XY[chosen_idxs, 0], XY[chosen_idxs, 1], s=45, alpha=0.95, label=f"Selected cluster (n={len(selected)})")

    # Centroid marker
    centroid = XY[chosen_idxs].mean(axis=0)
    ax.scatter([centroid[0]], [centroid[1]], s=220, marker="X", label="Selected centroid")

    # Ellipse boundary around selected points
    add_cov_ellipse(ax, XY[chosen_idxs], percentile=args.ellipse_percentile)

    # Report cluster stats (top by size)
    top = sorted(stats, key=lambda x: x[1], reverse=True)[:10]

    ax.set_title(
        f"LDA projection of embeddings + KMeans grouping (K={args.n_clusters})\n"
        f"Chosen cohesive cluster={chosen_label}, selected={len(selected)}  |  ellipse={args.ellipse_percentile:.0f}%"
    )
    ax.set_xlabel("LDA Component 1")
    ax.set_ylabel("LDA Component 2")
    ax.legend()
    ax.grid(True, alpha=0.2)

    fig.savefig(args.plot_png, dpi=200, bbox_inches="tight")

    print("✅ Done")
    print(f"Chosen label: {chosen_label}")
    print(f"Selected: {len(selected)} (target {args.target_min}-{args.target_max})")
    print("Top clusters (label, size, cohesion):", top)
    print(f"Saved FAQs: {args.out_jsonl}")
    print(f"Saved plot: {args.plot_png}")


if __name__ == "__main__":
    main()
