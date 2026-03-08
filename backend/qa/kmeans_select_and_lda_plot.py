# backend/qa/kmeans_select_and_lda_plot.py
import argparse
import json
from pathlib import Path
from typing import Dict, List

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


def cosine_dist_to_centroid(emb: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    X = emb[idxs]
    centroid = X.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
    sims = X @ centroid
    return 1.0 - sims


def choose_diverse_subset(labels: np.ndarray, emb: np.ndarray, target_total: int) -> List[int]:
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    
    base_alloc = target_total // k
    remainder = target_total % k
    
    counts_per_cluster = {lab: base_alloc for lab in unique_labels}
    
    cluster_sizes = {lab: np.sum(labels == lab) for lab in unique_labels}
    sorted_by_size = sorted(unique_labels, key=lambda x: cluster_sizes[x], reverse=True)
    for i in range(remainder):
        counts_per_cluster[sorted_by_size[i]] += 1
        
    selected_idxs = []
    for lab in unique_labels:
        idxs = np.where(labels == lab)[0]
        if len(idxs) == 0: continue
            
        dists = cosine_dist_to_centroid(emb, idxs)
        sorted_local = np.argsort(dists)
        
        n_to_take = min(counts_per_cluster[lab], len(idxs))
        selected_idxs.extend(idxs[sorted_local[:n_to_take]])
        
    return selected_idxs


def add_cov_ellipse(ax, pts: np.ndarray, color: str):
    """Draws a 95% confidence ellipse (circle) around a cluster of points."""
    if pts.shape[0] < 3:
        return
    mean = pts.mean(axis=0)
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    
    # 5.991 is the chi-square value for 95% confidence in 2D
    width = 2.0 * np.sqrt(vals[0] * 5.991)
    height = 2.0 * np.sqrt(vals[1] * 5.991)
    
    e = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                fill=False, linewidth=1.5, edgecolor=color, alpha=0.7)
    ax.add_patch(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--plot_png", required=True)
    ap.add_argument("--embed_model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--n_clusters", type=int, default=12)
    ap.add_argument("--target_total", type=int, default=50) # CHANGED TO 50

    args = ap.parse_args()

    rows = read_jsonl(Path(args.input_jsonl))
    rows = [r for r in rows if (r.get("question") or "").strip() and (r.get("answer") or "").strip()]

    questions = [r["question"].strip() for r in rows]
    model = SentenceTransformer(args.embed_model)
    emb = model.encode(questions, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)

    km = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(emb)

    chosen_idxs = choose_diverse_subset(labels, emb, args.target_total)
    selected = [rows[i] for i in chosen_idxs]
    write_jsonl(Path(args.out_jsonl), selected)

    lda = LDA(n_components=2)
    XY = lda.fit_transform(emb, labels)

    fig, ax = plt.subplots(figsize=(10, 7))

    mask = np.ones(len(rows), dtype=bool)
    mask[chosen_idxs] = False
    ax.scatter(XY[mask, 0], XY[mask, 1], c='gray', s=15, alpha=0.15, label="Discarded")

    cmap = plt.get_cmap('tab20')
    unique_chosen_labels = np.unique(labels[chosen_idxs])
    
    for i, lab in enumerate(unique_chosen_labels):
        idx_for_lab = [idx for idx in chosen_idxs if labels[idx] == lab]
        color = cmap(i % 20)
        
        # Plot the points
        ax.scatter(XY[idx_for_lab, 0], XY[idx_for_lab, 1], 
                   color=color, s=60, alpha=0.95, edgecolors='black')
        
        # Draw the "Circle" (Ellipse) around the points
        add_cov_ellipse(ax, XY[idx_for_lab], color=color)

    ax.set_title(f"LDA Projection with Cluster Boundaries\nTotal Selected: {len(selected)} FAQs")
    ax.set_xlabel("LDA Component 1")
    ax.set_ylabel("LDA Component 2")
    ax.grid(True, alpha=0.2)
    fig.savefig(args.plot_png, dpi=200, bbox_inches="tight")

    print(f"✅ Saved {len(selected)} FAQs and Plot with cluster boundaries!")

if __name__ == "__main__":
    main()