import io
import re
import sys
import math
import time
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit.runtime.scriptrunner import get_script_run_ctx

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from Bio import SeqIO


# =============================
# Utility & Core Helper Methods
# =============================

DNA_ALPHABET = set(list("ACGTUWSMKRYBDHVN-"))  # include ambiguous bases and gap
PROTEIN_ALPHABET = set(list("ACDEFGHIKLMNPQRSTVWYBXZ*-"))


def is_dna_sequence(seq: str, dna_threshold: float = 0.85) -> bool:
    """
    Heuristic to determine whether a sequence is DNA.
    - Count proportion of characters in DNA alphabet; if >= threshold -> DNA.
    - Empty returns False.
    """
    if not seq:
        return False
    seq = seq.upper()
    valid = sum(1 for c in seq if c in DNA_ALPHABET)
    ratio = valid / max(1, len(seq))
    return ratio >= dna_threshold


def parse_fasta(file_content: bytes) -> pd.DataFrame:
    """
    Parse uploaded FASTA bytes to a tidy DataFrame with columns:
      - id: sequence identifier (header without '>')
      - description: full header/description
      - sequence: raw sequence (uppercase, stripped)
      - type: 'DNA' or 'Protein'
      - taxon: extracted taxon if found in [ ... ] in header
    """
    handle = io.StringIO(file_content.decode(errors="ignore"))
    records = list(SeqIO.parse(handle, "fasta"))
    if not records:
        return pd.DataFrame(columns=["id", "description", "sequence", "type", "taxon"])  # empty

    rows = []
    for rec in records:
        seq = str(rec.seq).upper().replace("\n", "").replace("\r", "").strip()
        seq_type = "DNA" if is_dna_sequence(seq) else "Protein"
        taxon = extract_taxon_from_header(rec.description)
        rows.append({
            "id": rec.id,
            "description": rec.description,
            "sequence": seq,
            "type": seq_type,
            "taxon": taxon
        })

    return pd.DataFrame(rows)


TAXON_REGEXES = [
    re.compile(r"\[([^\]]+)\]"),             # [Homo sapiens]
    re.compile(r"\(([^\)]+)\)"),             # (Homo sapiens) as fallback
]


def extract_taxon_from_header(header: str) -> Optional[str]:
    """
    Extract taxonomic label from FASTA header.
    Strategy:
      1) Anything inside [ ... ] takes precedence.
      2) Fallback to ( ... ) if brackets not present.
      3) Return None if nothing plausible is found.
    """
    if not header:
        return None
    for rx in TAXON_REGEXES:
        m = rx.search(header)
        if m:
            label = m.group(1).strip()
            # Remove excessive whitespace
            label = re.sub(r"\s+", " ", label)
            return label if label else None
    return None


def generate_kmers(sequence: str, k: int) -> List[str]:
    """Generate overlapping k-mers from a sequence."""
    if not sequence:
        return []
    if k <= 0:
        return []
    kmers = [sequence[i:i + k] for i in range(max(0, len(sequence) - k + 1))]
    # Filter out kmers with gaps only or empty
    return [kmer for kmer in kmers if set(kmer) - set("- ")]


def sequences_to_kmer_tokens(df: pd.DataFrame, k_dna: int = 6, k_protein: int = 3) -> List[List[str]]:
    """
    Convert sequences to k-mer tokens per row, respecting type-specific k.
    Returns list of token lists aligned with df rows.
    """
    tokens: List[List[str]] = []
    for _, row in df.iterrows():
        k = k_dna if row["type"] == "DNA" else k_protein
        toks = generate_kmers(row["sequence"], k)
        tokens.append(toks)
    return tokens


def vectorize_tokens(tokens: List[List[str]], max_features: int = 5000) -> Tuple[np.ndarray, CountVectorizer]:
    """
    Vectorize token lists using Bag-of-k-mers via CountVectorizer.
    We pass identity tokenizer/preprocessor and disable token_pattern.
    Returns feature matrix (n_samples x n_features) and fitted vectorizer.
    """
    vectorizer = CountVectorizer(
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        token_pattern=None,
        max_features=max_features,
    )
    X = vectorizer.fit_transform(tokens)
    return X.toarray(), vectorizer


def reduce_dimensions(X: np.ndarray, method: str = "PCA", random_state: int = 42) -> np.ndarray:
    """
    Reduce feature space to 2D for visualization using PCA or t-SNE.
    PCA is fast; t-SNE is slower but can reveal non-linear structure.
    """
    if X.shape[0] == 0:
        return np.zeros((0, 2))

    if method == "TSNE":
        # Use PCA initialization for t-SNE to speed convergence on high-dim BOW.
        init = PCA(n_components=min(50, X.shape[1]), random_state=random_state).fit_transform(X)
        tsne = TSNE(n_components=2, init="pca", random_state=random_state, perplexity=min(30, max(5, X.shape[0] // 3)))
        pts = tsne.fit_transform(X)
    else:
        pca = PCA(n_components=2, random_state=random_state)
        pts = pca.fit_transform(X)
    return pts


def run_clustering(X: np.ndarray, algo: str, k: int = 5, eps: float = 0.5, min_samples: int = 5, random_state: int = 42) -> np.ndarray:
    """
    Cluster feature matrix using KMeans or DBSCAN.
    Returns cluster labels (shape: n_samples). Noise in DBSCAN is labeled -1.
    """
    if X.shape[0] == 0:
        return np.array([])
    if algo == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = model.fit_predict(X)
    else:
        k = max(2, min(k, max(2, X.shape[0] // 2)))
        model = KMeans(n_clusters=k, random_state=raw_seed(1))
        labels = model.fit_predict(X)
    return labels


def build_classifier(X: np.ndarray, y: List[Optional[str]], random_state: int = 42) -> Tuple[Optional[RandomForestClassifier], Dict[str, any]]:
    """
    Train a small RandomForest classifier to predict taxonomy labels from embeddings.
    Returns (model or None, metrics dict).
    """
    # Filter to rows with non-null labels
    labeled_indices = [i for i, lab in enumerate(y) if lab is not None and str(lab).strip() != ""]
    if len(labeled_indices) < 4:
        return None, {"message": "Insufficient labeled sequences (<4) for supervised demo."}

    X_lab = X[labeled_indices]
    y_lab = [y[i] for i in labeled_indices]

    # Need at least 2 classes
    if len(set(y_lab)) < 2:
        return None, {"message": "Only one taxon present; need >=2 classes to train."}

    # Stratified split when possible
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_lab, y_lab, test_size=0.25, random_state=random_state, stratify=y_lab
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X_lab, y_lab, test_size=0.25, random_state=random_state
        )

    clf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred, zero_division=0, output_dict=False)

    metrics = {
        "accuracy": acc,
        "confusion_matrix": cm,
        "report": report,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_classes": len(set(y_lab)),
    }
    return clf, metrics


def compute_cluster_abundance(labels: np.ndarray, weights: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Compute (weighted) counts per cluster and relative abundance."""
    if labels.size == 0:
        return pd.DataFrame(columns=["cluster", "count", "relative_abundance"])  # empty

    if weights is None:
        weights = np.ones_like(labels, dtype=float)

    df = pd.DataFrame({"cluster": labels, "w": weights})
    agg = df.groupby("cluster", as_index=False)["w"].sum().rename(columns={"w": "count"})
    total = agg["count"].sum()
    if total <= 0:
        agg["relative_abundance"] = 0.0
    else:
        agg["relative_abundance"] = agg["count"] / total
    # Sort with noise (-1) last
    agg["sort_key"] = agg["cluster"].apply(lambda c: (1, c) if c == -1 else (0, c))
    agg = agg.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)
    return agg


def summarize_known_vs_novel(labels: np.ndarray, taxa: List[Optional[str]]) -> Dict[str, any]:
    """
    A cluster is "Novel" if no sequences inside have a known taxon label.
    Returns summary dict with mapping cluster_id -> {status, n, n_known}.
    """
    if labels.size == 0:
        return {"clusters": {}, "n_clusters": 0, "n_novel": 0}
    df = pd.DataFrame({"cluster": labels, "taxon": taxa})
    groups = df.groupby("cluster")
    summary = {}
    n_novel = 0
    for cid, g in groups:
        n = len(g)
        n_known = g["taxon"].notna().sum()
        status = "Novel cluster" if n_known == 0 else "Known/Mixed"
        if status == "Novel cluster":
            n_novel += 1
        summary[int(cid)] = {"status": status, "n": int(n), "n_known": int(n_known)}
    return {"clusters": summary, "n_clusters": len(summary), "n_novel": n_novel}


def raw_seed(offset: int = 0) -> int:
    """Generate a pseudo-random but deterministic seed adjusted by offset."""
    base = 42
    return (base * 9973 + offset * 7919) % 2**31


# =====================
# Streamlit Application
# =====================

FASTA_FILE_ASSIGNMENT_LOCATIONS = [
    "app.py sidebar upload block: fasta_file = st.file_uploader(...)",
]


def debug_fasta_file(value):
    print("[DEBUG] fasta_file type:", type(value))
    print("[DEBUG] fasta_file value:", repr(value))
    print("[DEBUG] fasta_file assignment locations:", FASTA_FILE_ASSIGNMENT_LOCATIONS)


st.set_page_config(
    page_title="SeaQuence: AI Biodiversity Explorer",
    page_icon="🧬",
    layout="wide",
)

st.title("SeaQuence: AI Biodiversity Explorer")
st.caption("eDNA clustering, discovery of novel taxa, and lightweight classification — database-independent prototype")

with st.sidebar:
    st.header("Upload & Settings")
    fasta_file = st.file_uploader("Upload FASTA file", type=["fa", "fasta", "fna", "faa", "fas"]) 
    debug_fasta_file(fasta_file)
    if get_script_run_ctx() is None:
        sys.exit("This is a Streamlit app. Run it with: streamlit run app.py")

    meta_file = st.file_uploader("Optional metadata CSV (columns: id, weight or env vars)", type=["csv"]) 

    st.subheader("Tokenization")
    k_dna = st.number_input("k for DNA", min_value=3, max_value=10, value=6, step=1)
    k_prot = st.number_input("k for Protein", min_value=2, max_value=6, value=3, step=1)
    max_features = st.number_input("Max vocabulary (Bag-of-k-mers)", min_value=500, max_value=50000, value=5000, step=500)

    st.subheader("Clustering")
    algo = st.selectbox("Algorithm", ["KMeans", "DBSCAN"], index=0)
    k_clusters = st.slider("KMeans: number of clusters (k)", 2, 25, 8)
    eps = st.slider("DBSCAN: eps", 0.1, 5.0, 1.2, 0.1)
    min_samples = st.slider("DBSCAN: min_samples", 2, 50, 5)

    st.subheader("Embedding → 2D")
    dimred = st.selectbox("Method", ["PCA", "TSNE"], index=0, help="t-SNE is slower but can reveal non-linear structure.")

    st.subheader("Classification")
    enable_clf = st.checkbox("Train RandomForest on known taxa (PoC)", value=True)

    st.subheader("Abundance Weighting")
    weight_column = st.text_input("Metadata weight column (optional)", value="weight")


def load_metadata_weights(meta_bytes: Optional[bytes], seq_ids: List[str], weight_col: str) -> np.ndarray:
    if meta_bytes is None:
        return np.ones(len(seq_ids), dtype=float)
    try:
        dfm = pd.read_csv(io.BytesIO(meta_bytes))
    except Exception:
        return np.ones(len(seq_ids), dtype=float)
    if "id" not in dfm.columns:
        return np.ones(len(seq_ids), dtype=float)
    if weight_col not in dfm.columns:
        # If no weight col, try a heuristic from common env columns
        candidate = None
        for c in ["weight", "abundance", "depth", "reads", "count"]:
            if c in dfm.columns:
                candidate = c
                break
        if candidate is None:
            return np.ones(len(seq_ids), dtype=float)
        weight_col = candidate
    wmap = dict(zip(dfm["id"].astype(str), pd.to_numeric(dfm[weight_col], errors="coerce").fillna(0.0)))
    return np.array([wmap.get(str(sid), 1.0) for sid in seq_ids], dtype=float)


def plot_clusters_2d(points2d: np.ndarray, labels: np.ndarray, types: List[str]):
    if points2d.shape[0] == 0:
        st.info("No points to plot.")
        return
    dfp = pd.DataFrame({
        "x": points2d[:, 0],
        "y": points2d[:, 1],
        "cluster": labels,
        "type": types,
    })
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=dfp,
        x="x", y="y",
        hue="cluster",
        style="type",
        palette="tab10",
        s=60,
        alpha=0.8,
    )
    plt.title("Clusters in 2D ({})".format("t-SNE" if dimred == "TSNE" else "PCA"))
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    st.pyplot(plt.gcf(), clear_figure=True)


def plot_abundance(df_abund: pd.DataFrame):
    if df_abund.empty:
        st.info("No abundance data to display.")
        return
    plt.figure(figsize=(7, 4))
    sns.barplot(data=df_abund, x="cluster", y="relative_abundance", color="#4C78A8")
    plt.ylabel("Relative abundance")
    plt.xlabel("Cluster")
    plt.title("Cluster relative abundance")
    plt.gca().yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    st.pyplot(plt.gcf(), clear_figure=True)


def example_predictions_table(clf: RandomForestClassifier, X: np.ndarray, ids: List[str], taxa: List[Optional[str]], n: int = 10) -> pd.DataFrame:
    n = min(n, len(ids))
    if n == 0:
        return pd.DataFrame()
    # Sample without replacement for variety
    indices = list(range(len(ids)))
    random.Random(raw_seed(7)).shuffle(indices)
    take = indices[:n]
    pred = clf.predict(X[take])
    return pd.DataFrame({
        "id": [ids[i] for i in take],
        "true_taxon": [taxa[i] for i in take],
        "pred_taxon": pred,
    })


# ==================
# Main App Workflow
# ==================

tab1, tab2, tab3 = st.tabs(["Clustering", "Classification", "Abundance"])

if fasta_file is None:
    st.info("Upload a FASTA file to begin.")
    st.stop()

# Parse FASTA
with st.spinner("Parsing FASTA and typing sequences..."):
    df_seq = parse_fasta(fasta_file.read())

if df_seq.empty:
    st.error("No sequences found in the uploaded FASTA.")
    st.stop()

st.success(f"Parsed {len(df_seq)} sequences. DNA: {(df_seq['type']=='DNA').sum()}, Protein: {(df_seq['type']=='Protein').sum()}")

# Tokenize & Vectorize
with st.spinner("Tokenizing and vectorizing k-mers..."):
    token_lists = sequences_to_kmer_tokens(df_seq, k_dna=int(k_dna), k_protein=int(k_prot))
    X, vec = vectorize_tokens(token_lists, max_features=int(max_features))

# Dimensionality reduction
with st.spinner(f"Computing 2D projection via {dimred}..."):
    pts2d = reduce_dimensions(X, method=dimred)

# Clustering
with st.spinner("Clustering embeddings..."):
    labels = run_clustering(X, algo=algo, k=int(k_clusters), eps=float(eps), min_samples=int(min_samples))

summary = summarize_known_vs_novel(labels, df_seq["taxon"].tolist())

with tab1:
    st.subheader("Unsupervised Clustering")
    plot_clusters_2d(pts2d, labels, df_seq["type"].tolist())

    # Known vs Novel summary
    st.markdown("**Known taxa vs Novel clusters**")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Clusters", summary["n_clusters"])
    col_b.metric("Novel clusters", summary["n_novel"])
    col_c.metric("Noise (DBSCAN)", int((labels == -1).sum()))
    if summary["clusters"]:
        df_sum = pd.DataFrame([
            {"cluster": k, **v} for k, v in sorted(summary["clusters"].items(), key=lambda kv: kv[0])
        ])
        st.dataframe(df_sum, use_container_width=True)

with tab2:
    st.subheader("Proof-of-Concept Classification (RandomForest)")
    if enable_clf:
        clf, metrics = build_classifier(X, df_seq["taxon"].tolist())
        if clf is None:
            st.info(metrics.get("message", "Classifier not trained."))
        else:
            c1, c2, c3 = st.columns([2, 2, 3])
            c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            c2.metric("Classes", metrics["n_classes"]) 
            c3.metric("Test size", metrics["n_test"]) 

            with st.expander("Classification report"):
                st.text(metrics["report"])  # pretty text output

            with st.expander("Confusion matrix"):
                cm = metrics["confusion_matrix"]
                plt.figure(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.title("Confusion Matrix")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                st.pyplot(plt.gcf(), clear_figure=True)

            st.markdown("**Example predictions** (random subset)")
            ex = example_predictions_table(clf, X, df_seq["id"].tolist(), df_seq["taxon"].tolist(), n=10)
            if not ex.empty:
                st.dataframe(ex, use_container_width=True)
            else:
                st.info("Not enough data for examples.")
    else:
        st.info("Enable the classifier in the sidebar to train on known taxa.")

with tab3:
    st.subheader("Abundance Estimation")
    weights = load_metadata_weights(meta_file.read() if meta_file else None, df_seq["id"].astype(str).tolist(), weight_column.strip())
    df_abund = compute_cluster_abundance(labels, weights)
    plot_abundance(df_abund)

    with st.expander("Abundance table"):
        st.dataframe(df_abund, use_container_width=True, hide_index=True)


# ================
# Footer & Notes
# ================
st.markdown("---")
st.caption(
    "This prototype uses k-mer NLP-style embeddings for database-independent discovery, "
    "clustering to surface novel taxa, and a lightweight supervised model for proof-of-concept classification."
)


