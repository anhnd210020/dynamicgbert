"""
ETH-GBERT Dataset Analyser
Dataset: MulDiGraph.pkl
Usage: python analyse_dataset.py
"""

import pickle
import numpy as np
import os
import sys
from collections import Counter

# ── Optional imports (graceful fallback) ──────────────────────────────────────
try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    print("[WARN] networkx not installed. Graph-level stats will be skipped.")

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_PLT = True
except ImportError:
    HAS_PLT = False
    print("[WARN] matplotlib not installed. Plots will be skipped.")

# ══════════════════════════════════════════════════════════════════════════════
DATASET_PATH = "/home/ducanhhh/DynamicGBert/Dataset/MulDiGraph.pkl"
OUTPUT_DIR   = "/home/ducanhhh/DynamicGBert/Dataset/analysis_output"
# ══════════════════════════════════════════════════════════════════════════════

os.makedirs(OUTPUT_DIR, exist_ok=True)


def sep(title=""):
    print("\n" + "═" * 60)
    if title:
        print(f"  {title}")
        print("═" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────
sep("1. LOADING DATASET")
print(f"Path : {DATASET_PATH}")

with open(DATASET_PATH, "rb") as f:
    data = pickle.load(f)

print(f"Type : {type(data)}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. TOP-LEVEL STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
sep("2. TOP-LEVEL STRUCTURE")

if isinstance(data, dict):
    print(f"Keys ({len(data)}) : {list(data.keys())}")
    for k, v in data.items():
        print(f"  [{k}]  type={type(v).__name__}  ", end="")
        if hasattr(v, "shape"):
            print(f"shape={v.shape}  dtype={v.dtype}")
        elif isinstance(v, (list, tuple)):
            print(f"len={len(v)}")
        else:
            print(f"value={v}")

elif isinstance(data, (list, tuple)):
    print(f"Length : {len(data)}")
    print(f"Element[0] type : {type(data[0])}")

elif HAS_NX and isinstance(data, (nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph)):
    print("Dataset IS a NetworkX graph — jumping to graph analysis...")

else:
    print(f"Raw value: {data}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: detect the graph object
# ─────────────────────────────────────────────────────────────────────────────
def find_graph(obj):
    """Try common key names, or return obj itself if it is a graph."""
    if HAS_NX and isinstance(obj, (nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph)):
        return obj
    if isinstance(obj, dict):
        for key in ("graph", "G", "g", "network", "net", "multigraph"):
            if key in obj and HAS_NX and isinstance(obj[key], (nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph)):
                return obj[key]
    return None


G = find_graph(data)


# ─────────────────────────────────────────────────────────────────────────────
# 3. LABEL / CLASS DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
sep("3. LABEL / CLASS DISTRIBUTION")

labels = None

# Try to find labels in dict
if isinstance(data, dict):
    for key in ("y", "labels", "label", "target", "targets", "fraud"):
        if key in data:
            labels = np.array(data[key])
            print(f"Labels found under key '{key}'")
            break

# Try to find labels on graph nodes
if labels is None and G is not None:
    node_data = [d for _, d in G.nodes(data=True)]
    if node_data and "label" in node_data[0]:
        labels = np.array([d.get("label", -1) for _, d in G.nodes(data=True)])
        print("Labels found in node attributes ('label')")
    elif node_data and "y" in node_data[0]:
        labels = np.array([d.get("y", -1) for _, d in G.nodes(data=True)])
        print("Labels found in node attributes ('y')")

if labels is not None:
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    print(f"\nTotal samples : {total:,}")
    print(f"{'Class':<10} {'Count':>10} {'Ratio':>10}")
    print("-" * 32)
    for u, c in zip(unique, counts):
        label_name = "Fraud" if u == 1 else "Normal"
        print(f"  {u} ({label_name:<6}) {c:>10,} {c/total*100:>9.2f}%")
    fraud_count  = counts[list(unique).index(1)] if 1 in unique else 0
    normal_count = counts[list(unique).index(0)] if 0 in unique else 0
    if fraud_count and normal_count:
        ratio = normal_count / fraud_count
        print(f"\nNormal : Fraud ratio ≈ {ratio:.1f} : 1")
else:
    print("[INFO] No label array found at top level.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. NODE / EDGE FEATURES
# ─────────────────────────────────────────────────────────────────────────────
sep("4. NODE / EDGE FEATURES")

if isinstance(data, dict):
    for key in ("x", "node_features", "features", "feat", "X"):
        if key in data:
            arr = np.array(data[key])
            print(f"Node features key='{key}' : shape={arr.shape}  dtype={arr.dtype}")
            print(f"  min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}  std={arr.std():.4f}")
            print(f"  NaN count : {np.isnan(arr).sum()}")
            break

    for key in ("edge_index", "edges", "edge_attr", "adj"):
        if key in data:
            arr = np.array(data[key])
            print(f"Edge data key='{key}' : shape={arr.shape}  dtype={arr.dtype}")
            break


# ─────────────────────────────────────────────────────────────────────────────
# 5. NETWORKX GRAPH STATS
# ─────────────────────────────────────────────────────────────────────────────
if G is not None and HAS_NX:
    sep("5. NETWORKX GRAPH STATISTICS")

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"Graph type    : {type(G).__name__}")
    print(f"Nodes         : {n_nodes:,}")
    print(f"Edges         : {n_edges:,}")

    if n_nodes > 0:
        density = nx.density(G)
        print(f"Density       : {density:.6f}")

    # Degree stats
    degrees = [d for _, d in G.degree()]
    if degrees:
        print(f"\nDegree stats:")
        print(f"  min={min(degrees)}  max={max(degrees)}  mean={np.mean(degrees):.2f}  median={np.median(degrees):.2f}")

    # Connected components
    try:
        if G.is_directed():
            wcc = list(nx.weakly_connected_components(G))
            scc = list(nx.strongly_connected_components(G))
            print(f"\nWeakly connected components  : {len(wcc):,}")
            print(f"Strongly connected components: {len(scc):,}")
            largest_wcc = max(wcc, key=len)
            print(f"Largest WCC size             : {len(largest_wcc):,} nodes")
        else:
            cc = list(nx.connected_components(G))
            print(f"\nConnected components : {len(cc):,}")
            print(f"Largest component    : {len(max(cc, key=len)):,} nodes")
    except Exception as e:
        print(f"[WARN] Component analysis failed: {e}")

    # Node attribute names
    if G.number_of_nodes() > 0:
        sample_node = list(G.nodes(data=True))[0]
        print(f"\nNode attribute keys : {list(sample_node[1].keys())}")

    # Edge attribute names
    if G.number_of_edges() > 0:
        sample_edge = list(G.edges(data=True))[0]
        print(f"Edge attribute keys : {list(sample_edge[2].keys())}")

else:
    sep("5. GRAPH STATS")
    print("[INFO] No NetworkX graph object detected. Skipping graph stats.")


# ─────────────────────────────────────────────────────────────────────────────
# 6. SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
sep("6. SUMMARY (for teacher presentation)")

summary_lines = []

if G is not None and HAS_NX:
    summary_lines += [
        f"Dataset           : MulDiGraph (Ethereum Phishing Transaction Network)",
        f"Graph type        : {type(G).__name__}",
        f"Total nodes       : {G.number_of_nodes():,}",
        f"Total edges       : {G.number_of_edges():,}",
        f"Graph density     : {nx.density(G):.6f}",
    ]

if labels is not None:
    unique, counts = np.unique(labels, return_counts=True)
    fraud_count  = counts[list(unique).index(1)] if 1 in unique else 0
    normal_count = counts[list(unique).index(0)] if 0 in unique else 0
    summary_lines += [
        f"Total accounts    : {len(labels):,}",
        f"  - Normal        : {normal_count:,} ({normal_count/len(labels)*100:.1f}%)",
        f"  - Fraud/Phishing: {fraud_count:,} ({fraud_count/len(labels)*100:.1f}%)",
    ]

for line in summary_lines:
    print(f"  {line}")

# Save summary to txt
summary_path = os.path.join(OUTPUT_DIR, "dataset_summary.txt")
with open(summary_path, "w") as f:
    f.write("ETH-GBERT Dataset Analysis — MulDiGraph\n")
    f.write("=" * 60 + "\n")
    for line in summary_lines:
        f.write(line + "\n")
print(f"\n[INFO] Summary saved to: {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────────────────────────────────────────
if HAS_PLT:
    sep("7. GENERATING PLOTS")
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig)
    fig.suptitle("MulDiGraph Dataset — Analysis", fontsize=14, fontweight="bold")

    # Plot 1 — Class distribution
    if labels is not None:
        ax1 = fig.add_subplot(gs[0])
        unique, counts = np.unique(labels, return_counts=True)
        bar_labels = ["Normal" if u == 0 else "Fraud" for u in unique]
        colors = ["#4CAF50", "#F44336"][:len(unique)]
        bars = ax1.bar(bar_labels, counts, color=colors, edgecolor="black", width=0.5)
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                     f"{count:,}", ha="center", va="bottom", fontsize=9)
        ax1.set_title("Class Distribution")
        ax1.set_ylabel("Number of Accounts")
        ax1.set_xlabel("Class")

    # Plot 2 — Degree distribution (if graph)
    if G is not None and HAS_NX:
        ax2 = fig.add_subplot(gs[1])
        degrees = [d for _, d in G.degree()]
        ax2.hist(degrees, bins=50, color="#2196F3", edgecolor="black", log=True)
        ax2.set_title("Degree Distribution (log scale)")
        ax2.set_xlabel("Degree")
        ax2.set_ylabel("Count (log)")

    # Plot 3 — Pie chart of class ratio
    if labels is not None:
        ax3 = fig.add_subplot(gs[2])
        unique, counts = np.unique(labels, return_counts=True)
        pie_labels = ["Normal" if u == 0 else "Fraud" for u in unique]
        colors = ["#4CAF50", "#F44336"][:len(unique)]
        ax3.pie(counts, labels=pie_labels, colors=colors, autopct="%1.1f%%",
                startangle=140, wedgeprops={"edgecolor": "black"})
        ax3.set_title("Class Ratio")

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "dataset_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Plot saved to: {plot_path}")
    plt.show()
else:
    print("[INFO] Skipping plots (matplotlib not available).")

sep("DONE")
print(f"All outputs saved to: {OUTPUT_DIR}")