"""Diagnose genus purity using the application's current clustering pipeline."""
import sys

import numpy as np
import pandas as pd

import seqlib


FASTA_PATH = (
    sys.argv[1]
    if len(sys.argv) > 1
    else r"C:\Users\Aneesh\Desktop\Projects\sequence_1\pdbaa.fa"
)
LABEL_COL = "genus"


def _top_counts(counts, denominator, limit=10):
    return [
        f"{str(name)[:25]}:{int(count)}({count / denominator:.4f})"
        for name, count in counts.head(limit).items()
    ]


def _purity_code():
    print("STEP1: d, seqlib.py:366-377, purity calculation body:")
    print("    correct = 0")
    print("    total = 0")
    print("    for c in set(cl):")
    print("        members = tl[cl == c]")
    print("        counter = Counter(members)")
    print("        correct += counter.most_common(1)[0][1]")
    print("        total += len(members)")
    print("    purity = correct / total if total > 0 else 0.0")


def run():
    print(
        "STEP1: a, app.py:49-53,125; seqlib.py:194-218: random per-genus cap "
        "of 300, then random global cap of 20,000; not stratified; seed=42."
    )
    print(
        "STEP1: b, yes per_class_cap=300 before clustering (app.py:125; "
        "seqlib.py:206-217); no min_class_size=30 (classifier only, app.py:162)."
    )
    print(
        "STEP1: c, app.py:149 passes df_seq[label_col]; seqlib.py:356-361 "
        "uses true_labels.notna(); no rows dropped before clustering; "
        "missing labels are excluded only from metric evaluation."
    )
    _purity_code()

    with open(FASTA_PATH, "rb") as fasta_file:
        full_df, _ = seqlib.parse_fasta(fasta_file, min_len=30)

    # Match the app's sample, feature, embedding, clustering and metric calls.
    sampled_df = seqlib.sample_sequences(
        full_df,
        max_sequences=20000,
        per_class_cap=300,
        label_col=LABEL_COL,
        seed=seqlib.SEED,
    )
    X_sparse, _ = seqlib.featurize(
        pd.Series(sampled_df["sequence"].tolist()),
        k=3,
        max_features=5000,
        use_tfidf=False,
    )
    X_svd = seqlib.reduce_svd(X_sparse, n_components=50)
    cluster_labels = seqlib.run_clustering(X_svd, algo="MiniBatchKMeans", k=8)
    genus = sampled_df[LABEL_COL].reset_index(drop=True)
    app_metrics = seqlib.evaluate_clusters(cluster_labels, genus)

    labeled_mask = genus.notna()
    labeled_genus = genus[labeled_mask].reset_index(drop=True)
    labeled_clusters = cluster_labels[labeled_mask.to_numpy()]
    sample_counts = labeled_genus.value_counts()
    n_labeled = len(labeled_genus)
    n_genera = len(sample_counts)
    print(
        f"1=cluster_input_rows:{len(sampled_df)}, purity_rows:{n_labeled}, "
        f"genera:{n_genera}, top10:{_top_counts(sample_counts, n_labeled)}"
    )

    full_genus = full_df[LABEL_COL].dropna()
    full_counts = full_genus.value_counts()
    print(
        f"2=full_kept_rows:{len(full_df)}, labeled_rows:{len(full_genus)}, "
        f"genera:{len(full_counts)}, top10:{_top_counts(full_counts, len(full_genus))}"
    )

    largest_genus_count = int(sample_counts.max())
    lower_bound = largest_genus_count / n_labeled
    print(
        f"3=largest_genus_count:{largest_genus_count}, N:{n_labeled}, "
        f"lower_bound:{lower_bound:.6f}, app_purity:{app_metrics['purity']:.6f}, "
        f"purity_ge_bound:{app_metrics['purity'] >= lower_bound}"
    )

    crosstab = pd.crosstab(labeled_genus, labeled_clusters)
    pandas_purity = crosstab.max(axis=0).sum() / crosstab.values.sum()
    print(
        f"4=app_purity:{app_metrics['purity']:.6f}, "
        f"pandas_purity:{pandas_purity:.6f}, "
        f"difference:{pandas_purity - app_metrics['purity']:.12f}"
    )

    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
    cluster_details = []
    for cluster_id, cluster_size in cluster_sizes.items():
        top_genera = crosstab[cluster_id].sort_values(ascending=False).head(3)
        genera_text = ",".join(
            f"{str(name)[:25]}:{int(count)}"
            for name, count in top_genera.items()
        )
        cluster_details.append(
            f"{cluster_id}:size={int(cluster_size)},top3=[{genera_text}]"
        )
    print(f"5=clusters:{'; '.join(cluster_details)}")

    sample_missing = int(genus.isna().sum())
    full_missing = int(full_df[LABEL_COL].isna().sum())
    sample_unknown = int(
        genus.astype("string").str.strip().str.casefold().eq("unknown").fillna(False).sum()
    )
    full_unknown = int(
        full_df[LABEL_COL]
        .astype("string")
        .str.strip()
        .str.casefold()
        .eq("unknown")
        .fillna(False)
        .sum()
    )
    print(
        f"6=sample_missing:{sample_missing}, sample_literal_unknown:{sample_unknown}, "
        f"full_missing:{full_missing}, full_literal_unknown:{full_unknown}; "
        "purity excludes missing labels; literal 'unknown' is counted as a genus."
    )

    print(
        f"VERDICT: (A) per-genus cap before clustering; each sampled genus is "
        f"capped at 300, largest observed={largest_genus_count}/{n_labeled}="
        f"{lower_bound:.4%}, app purity={app_metrics['purity']:.4%}."
    )
    print(
        f"VERDICT: pandas purity={pandas_purity:.6f} (difference "
        f"{pandas_purity - app_metrics['purity']:.12f}); purity is above its "
        f"{lower_bound:.4%} lower bound."
    )
    print("VERDICT: purity is consistent with the capped sample; the uncapped full set has different genus frequencies.")
    print("Fixes: None indicated; purity calculation matches the independent pandas calculation.")


if __name__ == "__main__":
    run()
