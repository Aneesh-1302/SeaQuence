"""Diagnose what the app's default eight clusters capture."""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import seqlib


FASTA_PATH = r"C:\Users\Aneesh\Desktop\Projects\sequence_1\pdbaa.fa"
OUTPUT_DIR = r".\diag_out"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SEED = 42


def _family_label(description):
    first_header = description.split("; ", 1)[0]
    family = re.sub(
        r"^.*?\bChain\s+\w+\s*,\s*", "", first_header, flags=re.IGNORECASE
    )
    family = re.sub(r"\s*\[[^\]]*\]", "", family)
    return " ".join(family.lower().split())


def _distinct_3mers(sequence):
    return len({sequence[i : i + 3] for i in range(len(sequence) - 2)})


def _shuffle_metrics(cluster_labels, labels):
    cluster_labels = np.asarray(cluster_labels)
    labels = np.asarray(labels, dtype=object)
    mask = pd.notna(labels)
    true_labels = labels[mask]
    true_clusters = cluster_labels[mask]
    real = seqlib.evaluate_clusters(
        true_clusters, pd.Series(true_labels, dtype=object)
    )
    rng = np.random.RandomState(SEED)
    shuffled = []
    for _ in range(20):
        labels_shuffled = true_labels.copy()
        rng.shuffle(labels_shuffled)
        shuffled.append(
            seqlib.evaluate_clusters(
                true_clusters, pd.Series(labels_shuffled, dtype=object)
            )
        )
    result = {}
    for metric in ("ari", "nmi", "purity"):
        vals = [row[metric] for row in shuffled]
        result[metric] = (
            real[metric],
            float(np.mean(vals)),
            float(np.max(vals)),
        )
    return result


def _metric_text(metrics, name):
    real, mean, maximum = metrics[name]
    return f"{name}={real:.4f}/{mean:.4f}/{maximum:.4f}"


def run():
    with open(FASTA_PATH, "rb") as fasta_file:
        df, _ = seqlib.parse_fasta(fasta_file, min_len=30)
    df = seqlib.sample_sequences(
        df, max_sequences=20000, per_class_cap=300, label_col="genus", seed=SEED
    )
    sequences = df["sequence"].tolist()
    X_sparse, _ = seqlib.featurize(
        pd.Series(sequences), k=3, max_features=5000, use_tfidf=False
    )
    X_svd = seqlib.reduce_svd(X_sparse, n_components=50)
    cluster_labels = seqlib.run_clustering(
        X_svd, algo="MiniBatchKMeans", k=8
    )
    lengths = np.asarray([len(sequence) for sequence in sequences], dtype=float)
    distinct_3mers = np.asarray(
        [_distinct_3mers(sequence) for sequence in sequences], dtype=float
    )
    genus = df["genus"].to_numpy(dtype=object)
    families = np.asarray(
        [_family_label(description) for description in df["description"]],
        dtype=object,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    points_2d, plot_indices = seqlib.reduce_2d(
        X_sparse, method="SVD", sample_cap=5000
    )
    log_lengths = np.log10(lengths[plot_indices])
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        points_2d[:, 0], points_2d[:, 1], c=log_lengths, s=12, alpha=0.75
    )
    fig.colorbar(scatter, ax=ax, label="log10(sequence length)")
    ax.set_title("SVD by sequence length")
    fig.tight_layout()
    length_plot = os.path.join(OUTPUT_DIR, "svd_by_length.png")
    fig.savefig(length_plot, dpi=150)
    plt.close(fig)

    top_genera = pd.Series(genus).dropna().value_counts().head(8).index.tolist()
    sampled_genus = genus[plot_indices]
    fig, ax = plt.subplots(figsize=(8, 6))
    is_top = np.isin(sampled_genus, top_genera)
    ax.scatter(
        points_2d[~is_top, 0],
        points_2d[~is_top, 1],
        color="lightgrey",
        s=12,
        alpha=0.5,
    )
    colors = plt.get_cmap("tab10")
    for index, name in enumerate(top_genera):
        selected = sampled_genus == name
        ax.scatter(
            points_2d[selected, 0],
            points_2d[selected, 1],
            color=colors(index),
            s=12,
            alpha=0.8,
            label=str(name)[:25],
        )
    ax.set_title("SVD by genus (top 8)")
    ax.legend(fontsize="small", markerscale=1.5)
    fig.tight_layout()
    genus_plot = os.path.join(OUTPUT_DIR, "svd_by_genus.png")
    fig.savefig(genus_plot, dpi=150)
    plt.close(fig)
    print(f"A: {length_plot}, {genus_plot}")

    aa_fractions = {
        f"aa_{aa}": np.asarray(
            [sequence.count(aa) / len(sequence) for sequence in sequences]
        )
        for aa in AMINO_ACIDS
    }
    descriptors = {
        "log10_length": np.log10(lengths),
        "distinct_3mers": distinct_3mers,
        **aa_fractions,
    }
    component_top = []
    for component_index in range(2):
        correlations = []
        component = X_svd[:, component_index]
        for name, values in descriptors.items():
            pearson = pearsonr(component, values).statistic
            spearman = spearmanr(component, values).statistic
            score = max(abs(pearson), abs(spearman))
            correlations.append((score, name, pearson, spearman))
        correlations.sort(reverse=True)
        component_top.append(
            ",".join(
                f"{name}=P:{pearson:.3f},S:{spearman:.3f}"
                for _, name, pearson, spearman in correlations[:5]
            )
        )
    print(f"B: component1[{component_top[0]}]")
    print(f"B: component2[{component_top[1]}]")

    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
    cluster_statistics = {}
    for cluster_id in cluster_sizes.index:
        mask = cluster_labels == cluster_id
        cluster_statistics[cluster_id] = {
            "size": int(mask.sum()),
            "median_length": float(np.median(lengths[mask])),
            "median_3mers": float(np.median(distinct_3mers[mask])),
        }
        stats = cluster_statistics[cluster_id]
        print(
            f"B: cluster={cluster_id},size={stats['size']},"
            f"median_length={stats['median_length']:.1f},"
            f"median_distinct_3mers={stats['median_3mers']:.1f}"
        )

    family_counts = pd.Series(families).value_counts()
    top5_family_share = family_counts.head(5).sum() / len(families)
    print(
        f"C: families={len(family_counts)},top5_share={top5_family_share:.4f}"
    )
    genus_metrics = _shuffle_metrics(cluster_labels, genus)
    family_metrics = _shuffle_metrics(cluster_labels, families)
    print(
        f"C: genus {_metric_text(genus_metrics, 'nmi')}; "
        f"{_metric_text(genus_metrics, 'ari')}"
    )
    print(
        f"C: family {_metric_text(family_metrics, 'nmi')}; "
        f"{_metric_text(family_metrics, 'ari')}"
    )

    cluster_genus = pd.crosstab(pd.Series(genus), pd.Series(cluster_labels))
    cluster_family = pd.crosstab(pd.Series(families), pd.Series(cluster_labels))
    for cluster_id, stats in cluster_statistics.items():
        genus_counts = cluster_genus[cluster_id].sort_values(ascending=False)
        family_counts_in_cluster = cluster_family[cluster_id].sort_values(
            ascending=False
        )
        top_genus, genus_count = genus_counts.index[0], genus_counts.iloc[0]
        top_family, family_count = (
            family_counts_in_cluster.index[0],
            family_counts_in_cluster.iloc[0],
        )
        print(
            f"C: cluster={cluster_id},size={stats['size']},"
            f"family={str(top_family)[:25]}:{family_count / stats['size']:.3f},"
            f"genus={str(top_genus)[:25]}:{genus_count / stats['size']:.3f}"
        )
    family_score = family_metrics["nmi"][0]
    genus_score = genus_metrics["nmi"][0]
    better = "family" if family_score > genus_score else "genus"
    print(
        f"C: clusters line up better with {better} "
        f"(NMI family={family_score:.4f}, genus={genus_score:.4f})."
    )

    genus_chance = _shuffle_metrics(cluster_labels, genus)
    print(
        "D: chance levels "
        f"ARI={genus_chance['ari'][1]:.3f},"
        f"NMI={genus_chance['nmi'][1]:.3f},"
        f"Purity={genus_chance['purity'][1]:.3f}"
    )
    print(
        f"VERDICT: component1 top={component_top[0]}; "
        f"component2 top={component_top[1]}."
    )
    print(
        f"VERDICT: family NMI={family_score:.4f} vs genus NMI={genus_score:.4f}; "
        f"family top5 share={top5_family_share:.4f}."
    )


if __name__ == "__main__":
    run()
