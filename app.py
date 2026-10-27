"""
SeaQuence Streamlit UI — protein-only analysis.
All logic lives in seqlib.py; this file is UI only.
"""
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse

import seqlib

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SeaQuence: AI Biodiversity Explorer",
    page_icon="\U0001f9ec",
    layout="wide",
)
st.title("SeaQuence: AI Biodiversity Explorer")
st.caption(
    "Protein k-mer embeddings, unsupervised clustering, and lightweight classification — "
    "database-independent prototype"
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Upload & Settings")
    fasta_file = st.file_uploader(
        "Upload FASTA file", type=["fa", "fasta", "fna", "faa", "fas"]
    )
    meta_file = st.file_uploader(
        "Optional metadata CSV (columns: id, weight)", type=["csv"]
    )

    st.subheader("Tokenization")
    k_prot = st.number_input("k for protein k-mers", min_value=2, max_value=6, value=3, step=1)
    max_features = st.number_input(
        "Max vocabulary", min_value=500, max_value=50000, value=5000, step=500
    )
    use_tfidf = st.checkbox("Use TF-IDF (instead of L1 normalization)", value=False)

    st.subheader("Sampling")
    max_sequences = st.number_input(
        "Max sequences (random sample)", min_value=100, max_value=500000, value=20000, step=1000
    )
    per_class_cap = st.number_input(
        "Per-class cap", min_value=10, max_value=10000, value=300, step=50
    )

    st.subheader("Clustering")
    algo = st.selectbox("Algorithm", ["MiniBatchKMeans", "DBSCAN"], index=0)
    k_clusters = st.slider("KMeans: number of clusters (k)", 2, 25, 8)
    eps = st.slider("DBSCAN: eps", 0.1, 5.0, 1.2, 0.1)
    min_samples = st.slider("DBSCAN: min_samples", 2, 50, 5)

    st.subheader("Embedding 2D")
    dimred = st.selectbox(
        "Method", ["SVD", "TSNE"], index=0,
        help="t-SNE is slower; applied to a random sample of at most 5 000 points.",
    )

    st.subheader("Classification")
    enable_clf = st.checkbox("Train classifiers on genus labels", value=True)
    label_col = st.selectbox("Label level", ["genus", "organism"], index=0)
    min_class_size = st.number_input(
        "Min examples per class", min_value=2, max_value=500, value=30, step=5
    )

    st.subheader("Abundance Weighting")
    weight_column = st.text_input("Metadata weight column", value="weight")

    run_button = st.button("\U0001f680 Run analysis", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# State gate
# ---------------------------------------------------------------------------
if fasta_file is None:
    st.info("Upload a FASTA file to begin.")
    st.stop()

if not run_button and "results" not in st.session_state:
    st.info("Configure parameters, then press **Run analysis**.")
    st.stop()

# ---------------------------------------------------------------------------
# Cached heavy functions
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Parsing FASTA…")
def cached_parse(data: bytes, min_len: int):
    return seqlib.parse_fasta(data, min_len=min_len)


@st.cache_data(show_spinner="Building k-mer features…")
def cached_featurize(sequences_list, k, max_feat, tfidf):
    seqs = pd.Series(sequences_list)
    X, vec = seqlib.featurize(seqs, k=k, max_features=max_feat, use_tfidf=tfidf)
    return X, vec


@st.cache_data(show_spinner=False)
def cached_chance_metrics(cluster_labels, true_labels):
    labels = np.asarray(true_labels, dtype=object)
    clusters = np.asarray(cluster_labels)
    mask = pd.notna(labels)
    labels, clusters = labels[mask], clusters[mask]
    rng = np.random.RandomState(seqlib.SEED)
    scores = [seqlib.evaluate_clusters(clusters, pd.Series(rng.permutation(labels))) for _ in range(20)]
    return {key: float(np.mean([score[key] for score in scores])) for key in ("ari", "nmi", "purity")}


# ---------------------------------------------------------------------------
# Pipeline (runs on button press)
# ---------------------------------------------------------------------------
if run_button:
    raw = fasta_file.read()
    df_seq, audit = cached_parse(raw, int(30))

    if df_seq.empty:
        st.error("No sequences survived filtering.")
        st.stop()

    # Audit display
    with st.expander("Parsing audit", expanded=True):
        for key in ["records", "kept", "ambiguous", "viruses", "too_short",
                     "bytes_replaced", "unique_genus", "unique_species",
                     "length_median", "length_min", "length_max"]:
            st.write(f"**{key}**: {audit.get(key, '—')}")

    # Sample
    df_seq = seqlib.sample_sequences(
        df_seq, max_sequences=int(max_sequences),
        per_class_cap=int(per_class_cap), label_col=label_col,
    )
    st.success(f"Working with {len(df_seq)} sequences after sampling.")

    # Featurize
    X_sparse, vec = cached_featurize(
        df_seq["sequence"].tolist(), int(k_prot), int(max_features), use_tfidf,
    )

    # SVD 50-D
    X_svd = seqlib.reduce_svd(X_sparse, n_components=50)

    # 2D projection
    pts2d, plot_idx = seqlib.reduce_2d(X_sparse, method=dimred)

    # Clustering on 50-D
    cluster_labels = seqlib.run_clustering(
        X_svd, algo=algo, k=int(k_clusters),
        eps=float(eps), min_samples=int(min_samples),
    )

    # Evaluate clusters
    cluster_metrics = seqlib.evaluate_clusters(cluster_labels, df_seq[label_col])

    # Known vs unannotated
    kv_summary = seqlib.summarize_known_vs_unannotated(
        cluster_labels, df_seq[label_col].tolist(),
    )

    # Classification
    clf_results = {}
    if enable_clf:
        clf_results = seqlib.run_classification(
            X_sparse, X_svd, df_seq,
            label_col=label_col,
            min_class_size=int(min_class_size),
            per_class_cap=int(per_class_cap),
        )

    # Abundance
    meta_bytes = meta_file.read() if meta_file else None
    weights, n_matched = seqlib.load_metadata_weights(
        meta_bytes, df_seq["id"].astype(str).tolist(), weight_column.strip(),
    )
    df_abund = seqlib.compute_cluster_abundance(cluster_labels, weights)

    # Run metadata JSON
    run_meta = seqlib.build_run_metadata(
        k=int(k_prot), max_features=int(max_features),
        max_sequences=int(max_sequences), per_class_cap=int(per_class_cap),
        min_class_size=int(min_class_size), label_col=label_col,
        use_tfidf=use_tfidf, audit=audit,
        cluster_metrics=cluster_metrics, classification_results=clf_results,
    )

    # Store everything in session state
    st.session_state["results"] = {
        "df_seq": df_seq,
        "audit": audit,
        "pts2d": pts2d,
        "plot_idx": plot_idx,
        "cluster_labels": cluster_labels,
        "cluster_metrics": cluster_metrics,
        "kv_summary": kv_summary,
        "clf_results": clf_results,
        "df_abund": df_abund,
        "weights": weights,
        "n_matched": n_matched,
        "run_meta": run_meta,
        "dimred": dimred,
        "label_col": label_col,
    }

# ---------------------------------------------------------------------------
# Display (from session state)
# ---------------------------------------------------------------------------
if "results" not in st.session_state:
    st.stop()

R = st.session_state["results"]
df_seq = R["df_seq"]
pts2d = R["pts2d"]
plot_idx = R["plot_idx"]
cluster_labels = R["cluster_labels"]
cluster_metrics = R["cluster_metrics"]
kv_summary = R["kv_summary"]
clf_results = R["clf_results"]
df_abund = R["df_abund"]
run_meta = R["run_meta"]
dimred_used = R["dimred"]
label_col_used = R["label_col"]

tab1, tab2, tab3, tab4 = st.tabs(["Clustering", "Classification", "Abundance", "Run metadata"])

# ------ TAB 1: Clustering ------
with tab1:
    st.subheader("Unsupervised Clustering")

    # Cluster evaluation metrics
    mc1, mc2, mc3 = st.columns(3)
    chance_metrics = cached_chance_metrics(tuple(cluster_labels), tuple(df_seq[label_col_used]))
    mc1.metric("ARI", f"{cluster_metrics.get('ari', 0):.3f}")
    mc1.caption(f"chance level ~ {chance_metrics['ari']:.3f}")
    mc2.metric("NMI", f"{cluster_metrics.get('nmi', 0):.3f}")
    mc2.caption(f"chance level ~ {chance_metrics['nmi']:.3f}")
    mc3.metric("Purity", f"{cluster_metrics.get('purity', 0):.3f}")
    mc3.caption(f"chance level ~ {chance_metrics['purity']:.3f}")
    st.caption(f"Evaluated on {cluster_metrics.get('n_labeled', 0)} labeled sequences.")

    # 2D scatter
    if pts2d.shape[0] > 0:
        plot_labels = cluster_labels[plot_idx] if len(plot_idx) < len(cluster_labels) else cluster_labels
        dfp = pd.DataFrame({
            "x": pts2d[:, 0],
            "y": pts2d[:, 1],
            "cluster": plot_labels.astype(str),
        })
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=dfp, x="x", y="y", hue="cluster", palette="tab10", s=40, alpha=0.7, ax=ax)
        ax.set_title(f"Clusters in 2D ({dimred_used})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=8)
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("No points to plot.")

    # Known vs Unannotated summary
    st.markdown("**Known taxa vs Unannotated clusters**")
    st.caption("An unannotated cluster is NOT evidence of a new taxon — it only means no labeled sequences fell into that cluster.")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Clusters", kv_summary["n_clusters"])
    col_b.metric("Unannotated clusters", kv_summary["n_unannotated"])
    col_c.metric("Noise (DBSCAN)", int((cluster_labels == -1).sum()))
    if kv_summary["clusters"]:
        df_sum = pd.DataFrame([
            {"cluster": k, **v}
            for k, v in sorted(kv_summary["clusters"].items(), key=lambda kv2: kv2[0])
        ])
        st.dataframe(df_sum.head(20), use_container_width=True)

# ------ TAB 2: Classification ------
with tab2:
    st.subheader(f"Classification (label: {label_col_used})")
    if not clf_results or "error" in clf_results:
        st.info(clf_results.get("error", "Classification not run.") if clf_results else "Enable classification in sidebar.")
    else:
        majority_base = clf_results.get("majority_baseline", 0)
        st.metric("Majority baseline accuracy", f"{majority_base:.3f}")

        for split_name, models in clf_results.items():
            if not isinstance(models, list):
                continue
            st.markdown(f"---\n### {split_name}")
            for m in models:
                with st.expander(f"{m['name']}  —  acc {m['accuracy']:.3f} / bal {m['balanced_accuracy']:.3f} / F1 {m['macro_f1']:.3f}"):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Accuracy", f"{m['accuracy']:.3f}")
                    c2.metric("Balanced acc", f"{m['balanced_accuracy']:.3f}")
                    c3.metric("Macro F1", f"{m['macro_f1']:.3f}")
                    c4.metric("Classes", m["n_classes"])

                    st.text(m["report"])

                    # Confusion matrix
                    cm = m["confusion_matrix"]
                    cm_labels = m["cm_labels"]
                    if cm.shape[0] > 0:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(
                            cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=cm_labels, yticklabels=cm_labels, ax=ax,
                        )
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("True")
                        ax.set_title(f"Confusion matrix (top {len(cm_labels)} classes)")
                        plt.xticks(rotation=45, ha="right", fontsize=7)
                        plt.yticks(fontsize=7)
                        st.pyplot(fig, clear_figure=True)

                    # Example predictions (from test set only)
                    st.markdown("**Example predictions (held-out test set)**")
                    n_ex = min(10, len(m["y_test"]))
                    if n_ex > 0:
                        ex_idx = list(range(n_ex))
                        ex_df = pd.DataFrame({
                            "true": [m["y_test"][i] for i in ex_idx],
                            "predicted": [m["predictions"][i] for i in ex_idx],
                        })
                        st.dataframe(ex_df, use_container_width=True)

# ------ TAB 3: Abundance ------
with tab3:
    st.subheader("Cluster Abundance")
    st.caption(
        "Relative cluster counts, not organism abundance. "
        f"Metadata weights matched {R['n_matched']} / {len(df_seq)} sequence IDs."
    )
    if not df_abund.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=df_abund, x="cluster", y="relative_abundance", color="#4C78A8", ax=ax)
        ax.set_ylabel("Relative abundance")
        ax.set_xlabel("Cluster")
        ax.set_title("Cluster relative abundance")
        st.pyplot(fig, clear_figure=True)
        with st.expander("Abundance table"):
            st.dataframe(df_abund, use_container_width=True, hide_index=True)
    else:
        st.info("No abundance data.")

# ------ TAB 4: Run metadata ------
with tab4:
    st.subheader("Run Metadata")
    run_json = json.dumps(run_meta, indent=2, default=str)
    st.download_button(
        "Download run metadata (JSON)",
        data=run_json,
        file_name="seaquence_run.json",
        mime="application/json",
    )
    st.json(run_meta)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "SeaQuence uses protein k-mer embeddings for database-independent discovery, "
    "clustering to surface unannotated groups, and lightweight classifiers. "
    "Unannotated clusters are not evidence of new taxa."
)
