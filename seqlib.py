"""
seqlib.py - Pure logic for SeaQuence (protein-only).
Parsing, cleaning, feature extraction, dimensionality reduction,
clustering, classification, evaluation, and abundance utilities.
"""
import io
import re
import json
import random
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report,
    adjusted_rand_score, normalized_mutual_info_score,
)
from sklearn.preprocessing import normalize as sklearn_normalize
from Bio import SeqIO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
INVALID_AA = set("XBZ*")
VIRUS_RE = re.compile(r"virus$|phage", re.IGNORECASE)
ORGANISM_RE = re.compile(r"\[([^\]]+)\]")


# ---------------------------------------------------------------------------
# Parsing & Cleaning
# ---------------------------------------------------------------------------

def _normalize_organism(name: str) -> str:
    """Reduce to genus + species: strip strain, serovar, subsp."""
    if not name:
        return name
    name = re.sub(r"\s+(str\.|strain|serovar|subsp\.?)\s+.*", "", name)
    words = name.split()
    if len(words) >= 2:
        return words[0] + " " + words[1]
    return name.strip()


def _genus(species: str) -> Optional[str]:
    if not species:
        return None
    return species.split()[0]


def _extract_protein_name(description: str) -> str:
    """Extract protein name from first sub-header for grouping."""
    first = description.split("\x01")[0]
    # Format: "ID Chain X, PROTEIN NAME [Organism]"
    m = re.search(r"Chain\s+\w+,\s*(.+?)\s*\[", first)
    if m:
        return m.group(1).strip().upper()
    # Fallback: everything between first comma and bracket
    m2 = re.search(r",\s*(.+?)\s*\[", first)
    if m2:
        return m2.group(1).strip().upper()
    return first.strip().upper()


def parse_fasta(file_handle, *, min_len: int = 30) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Stream-parse a FASTA file handle (binary or text).
    Returns (DataFrame, audit_dict).
    """
    # Handle bytes vs file-like
    if isinstance(file_handle, bytes):
        replaced = file_handle.count(b"\xef\xbf\xbd")  # replacement char in UTF-8
        text = file_handle.decode("utf-8", errors="replace")
        replaced += text.count("\ufffd")
        text = text.replace("\r", "")
        handle = io.StringIO(text)
    else:
        raw = file_handle.read()
        if isinstance(raw, bytes):
            replaced = 0
            text = raw.decode("utf-8", errors="replace")
            replaced = text.count("\ufffd")
            text = text.replace("\r", "")
            handle = io.StringIO(text)
        else:
            replaced = 0
            text = raw.replace("\r", "")
            handle = io.StringIO(text)

    audit: Dict[str, Any] = {
        "records": 0,
        "kept": 0,
        "ambiguous": 0,
        "viruses": 0,
        "too_short": 0,
        "bytes_replaced": replaced,
        "unique_genus": 0,
        "unique_species": 0,
        "length_median": 0,
        "length_min": 0,
        "length_max": 0,
        "genus_counts": {},
        "species_counts": {},
    }

    rows = []
    for rec in SeqIO.parse(handle, "fasta"):
        audit["records"] += 1
        seq = str(rec.seq).upper().replace("\n", "").replace("\r", "").strip()
        desc = rec.description.replace("\r", "")
        rec_id = rec.id.replace("\r", "")

        # Split on \x01, extract organisms from each sub-header
        sub_headers = desc.split("\x01")
        organisms_raw = []
        for sh in sub_headers:
            m = ORGANISM_RE.search(sh)
            organisms_raw.append(m.group(1).strip() if m else None)

        # Virus filter: check raw organisms BEFORE normalization
        raw_virus = any(
            o and VIRUS_RE.search(o) for o in organisms_raw
        )
        if raw_virus:
            audit["viruses"] += 1
            continue

        # Normalize all organisms
        organisms_norm = [_normalize_organism(o) if o else None for o in organisms_raw]
        unique_orgs = set(o for o in organisms_norm if o is not None)

        if len(unique_orgs) > 1:
            audit["ambiguous"] += 1
            continue

        organism = unique_orgs.pop() if unique_orgs else None

        # Clean description: replace \x01 with '; '
        desc = desc.replace("\x01", "; ")

        # Length filter
        if len(seq) < min_len:
            audit["too_short"] += 1
            continue

        genus = _genus(organism)
        protein_name = _extract_protein_name(desc)

        rows.append({
            "id": rec_id,
            "description": desc,
            "sequence": seq,
            "organism": organism,
            "genus": genus,
            "protein_name": protein_name,
            "seq_len": len(seq),
        })

    df = pd.DataFrame(rows)
    audit["kept"] = len(df)

    if not df.empty:
        lengths = df["seq_len"].values
        audit["length_median"] = int(np.median(lengths))
        audit["length_min"] = int(lengths.min())
        audit["length_max"] = int(lengths.max())

        genus_vc = df["genus"].dropna().value_counts()
        species_vc = df["organism"].dropna().value_counts()
        audit["unique_genus"] = len(genus_vc)
        audit["unique_species"] = len(species_vc)
        audit["genus_counts"] = genus_vc.to_dict()
        audit["species_counts"] = species_vc.to_dict()

    return df, audit


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_sequences(
    df: pd.DataFrame,
    max_sequences: int = 20000,
    per_class_cap: int = 300,
    label_col: str = "genus",
    seed: int = SEED,
) -> pd.DataFrame:
    """Cap total and per-class counts via random sampling."""
    if df.empty:
        return df
    rng = np.random.RandomState(seed)

    # Per-class cap first
    if per_class_cap > 0 and label_col in df.columns:
        parts = []
        for _, group in df.groupby(label_col, dropna=False):
            if len(group) > per_class_cap:
                group = group.sample(per_class_cap, random_state=rng)
            parts.append(group)
        df = pd.concat(parts, ignore_index=True)

    # Global cap
    if len(df) > max_sequences:
        df = df.sample(max_sequences, random_state=rng)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def featurize(
    sequences: pd.Series,
    k: int = 3,
    max_features: int = 5000,
    use_tfidf: bool = False,
) -> Tuple[sparse.csr_matrix, CountVectorizer]:
    """
    Build sparse k-mer feature matrix.
    Ignores k-mers containing X, B, Z, or *.
    Returns (sparse float32 matrix, fitted vectorizer).
    """
    vec = CountVectorizer(
        analyzer="char",
        ngram_range=(k, k),
        lowercase=False,
        max_features=max_features,
    )
    X = vec.fit_transform(sequences)

    # Remove columns whose feature name contains invalid AA
    feature_names = vec.get_feature_names_out()
    keep_mask = np.array(
        [not any(c in INVALID_AA for c in fn) for fn in feature_names]
    )
    if not keep_mask.all():
        X = X[:, keep_mask]
        # Update the vectorizer vocabulary
        kept_names = feature_names[keep_mask]
        vec.vocabulary_ = {name: i for i, name in enumerate(kept_names)}

    X = X.astype(np.float32)

    if use_tfidf:
        X = TfidfTransformer().fit_transform(X)
    else:
        X = sklearn_normalize(X, norm="l1", copy=False)

    return X, vec


# ---------------------------------------------------------------------------
# Dimensionality Reduction
# ---------------------------------------------------------------------------

def reduce_svd(X: sparse.csr_matrix, n_components: int = 50) -> np.ndarray:
    """TruncatedSVD on sparse matrix."""
    if X.shape[0] == 0:
        return np.zeros((0, n_components))
    n = min(n_components, X.shape[1] - 1, X.shape[0] - 1)
    if n < 1:
        n = 1
    svd = TruncatedSVD(n_components=n, random_state=SEED)
    return svd.fit_transform(X)


def reduce_2d(X, method: str = "SVD", sample_cap: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce to 2D for plotting.
    Returns (points_2d, sample_indices).
    Always plots at most sample_cap points.
    """
    n = X.shape[0]
    if n == 0:
        return np.zeros((0, 2)), np.array([], dtype=int)

    if n > sample_cap:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(n, sample_cap, replace=False)
    else:
        idx = np.arange(n)

    X_sub = X[idx]

    if method == "TSNE":
        if n < 10:
            return np.zeros((len(idx), 2)), idx
        perp = min(30, max(5, len(idx) // 3 - 1))
        # Use SVD first for dense init
        if sparse.issparse(X_sub):
            X_dense = reduce_svd(X_sub, n_components=min(50, X_sub.shape[1] - 1))
        else:
            X_dense = X_sub
        tsne = TSNE(n_components=2, init="pca", random_state=SEED, perplexity=perp)
        pts = tsne.fit_transform(X_dense)
    else:
        if sparse.issparse(X_sub):
            nc = min(2, X_sub.shape[1] - 1, X_sub.shape[0] - 1)
            if nc < 2:
                nc = min(2, X_sub.shape[1], X_sub.shape[0])
            svd = TruncatedSVD(n_components=min(2, nc), random_state=SEED)
            pts = svd.fit_transform(X_sub)
            if pts.shape[1] < 2:
                pts = np.column_stack([pts, np.zeros(pts.shape[0])])
        else:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=SEED)
            pts = pca.fit_transform(X_sub)

    return pts, idx


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def run_clustering(
    X: np.ndarray,
    algo: str = "MiniBatchKMeans",
    k: int = 5,
    eps: float = 0.5,
    min_samples: int = 5,
) -> np.ndarray:
    """Cluster on dense embeddings (e.g., 50-D SVD)."""
    n = X.shape[0]
    if n < 3:
        print(f"Insufficient samples for clustering (n_samples={n} < 3). Skipping clustering.")
        return np.zeros(n, dtype=int)

    if algo == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        return model.fit_predict(X)
    else:
        k = max(2, min(k, n - 1))
        model = MiniBatchKMeans(n_clusters=k, random_state=SEED, batch_size=min(1000, n), n_init=3)
        return model.fit_predict(X)


def evaluate_clusters(
    labels: np.ndarray, true_labels: pd.Series
) -> Dict[str, float]:
    """ARI, NMI, purity on labeled subset."""
    mask = true_labels.notna()
    if mask.sum() < 2:
        return {"ari": float("nan"), "nmi": float("nan"), "purity": float("nan"), "n_labeled": int(mask.sum())}

    cl = labels[mask]
    tl = true_labels[mask].values

    ari = adjusted_rand_score(tl, cl)
    nmi = normalized_mutual_info_score(tl, cl)

    # Purity
    from collections import Counter
    total = 0
    correct = 0
    for c in set(cl):
        members = tl[cl == c]
        counter = Counter(members)
        correct += counter.most_common(1)[0][1]
        total += len(members)
    purity = correct / total if total > 0 else 0.0

    return {"ari": ari, "nmi": nmi, "purity": purity, "n_labeled": int(mask.sum())}


# ---------------------------------------------------------------------------
# Known vs Unannotated
# ---------------------------------------------------------------------------

def summarize_known_vs_unannotated(
    labels: np.ndarray, taxa: List[Optional[str]]
) -> Dict[str, Any]:
    """
    A cluster is 'Unannotated' if no sequences inside have a known taxon label.
    This is NOT evidence of a new taxon.
    """
    if labels.size == 0:
        return {"clusters": {}, "n_clusters": 0, "n_unannotated": 0}
    df = pd.DataFrame({"cluster": labels, "taxon": taxa})
    groups = df.groupby("cluster")
    summary = {}
    n_unannotated = 0
    for cid, g in groups:
        n = len(g)
        n_known = g["taxon"].notna().sum()
        status = "Unannotated cluster" if n_known == 0 else "Known/Mixed"
        if status == "Unannotated cluster":
            n_unannotated += 1
        summary[int(cid)] = {"status": status, "n": int(n), "n_known": int(n_known)}
    return {"clusters": summary, "n_clusters": len(summary), "n_unannotated": n_unannotated}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _prepare_classification_data(
    X,
    df: pd.DataFrame,
    label_col: str = "genus",
    min_class_size: int = 30,
    per_class_cap: int = 300,
) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter to classes with >= min_class_size, cap each class.
    Returns (X_filtered, y, groups_for_split, indices).
    """
    labels = df[label_col].values
    mask = pd.notna(labels)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return None, np.array([]), np.array([]), np.array([])

    y = labels[indices]
    # Filter by min_class_size
    counts = pd.Series(y).value_counts()
    valid_classes = set(counts[counts >= min_class_size].index)
    if len(valid_classes) < 2:
        return None, np.array([]), np.array([]), np.array([])

    keep = np.array([v in valid_classes for v in y])
    indices = indices[keep]
    y = y[keep]

    # Per-class cap
    if per_class_cap > 0:
        rng = np.random.RandomState(SEED)
        final_idx = []
        for cls in valid_classes:
            cls_positions = np.where(y == cls)[0]
            if len(cls_positions) > per_class_cap:
                cls_positions = rng.choice(cls_positions, per_class_cap, replace=False)
            final_idx.extend(cls_positions)
        final_idx = sorted(final_idx)
        indices = indices[final_idx]
        y = y[final_idx]

    X_filt = X[indices]

    # Groups: protein_name + genus
    groups = np.array([
        df.iloc[i]["protein_name"] + "||" + str(df.iloc[i][label_col])
        for i in indices
    ])

    return X_filt, y, groups, indices


def _train_eval_model(model, X_train, X_test, y_train, y_test, name: str) -> Dict[str, Any]:
    """Train and evaluate a single model."""
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    classes = sorted(set(y_test) | set(y_train))
    acc = accuracy_score(y_test, pred)
    bal_acc = balanced_accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    report = classification_report(
        y_test, pred, labels=classes, target_names=classes,
        zero_division=0, output_dict=False,
    )
    # Confusion matrix for top 15 classes by frequency in test set
    test_counts = pd.Series(y_test).value_counts()
    top15 = list(test_counts.head(15).index)
    cm_mask_test = np.isin(y_test, top15)
    cm_mask_pred = np.isin(pred, top15)  # only used for labeling
    cm = confusion_matrix(y_test[cm_mask_test], pred[cm_mask_test], labels=top15)

    return {
        "name": name,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "report": report,
        "confusion_matrix": cm,
        "cm_labels": top15,
        "predictions": pred,
        "y_test": y_test,
        "X_test": X_test,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_classes": len(classes),
    }


def run_classification(
    X_sparse,
    X_svd: np.ndarray,
    df: pd.DataFrame,
    label_col: str = "genus",
    min_class_size: int = 30,
    per_class_cap: int = 300,
) -> Dict[str, Any]:
    """
    Run classification with grouped and random splits.
    Returns dict with results per split type per model.
    """
    X_filt, y, groups, indices = _prepare_classification_data(
        X_sparse, df, label_col, min_class_size, per_class_cap,
    )
    if X_filt is None or len(y) < 4 or len(set(y)) < 2:
        return {"error": "Insufficient labeled data for classification."}

    X_svd_filt = X_svd[indices]

    results = {}

    # Majority baseline accuracy (on full set)
    majority_class = pd.Series(y).value_counts().index[0]
    majority_acc = (y == majority_class).mean()
    results["majority_baseline"] = majority_acc

    split_configs = []

    # 1. Grouped split
    try:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
        train_idx, test_idx = next(gss.split(X_filt, y, groups))
        split_configs.append(("grouped split", train_idx, test_idx))
    except ValueError:
        pass

    # 2. Random split
    all_idx = np.arange(len(y))
    try:
        train_idx_r, test_idx_r = train_test_split(
            all_idx, test_size=0.25, random_state=SEED, stratify=y,
        )
    except ValueError:
        train_idx_r, test_idx_r = train_test_split(
            all_idx, test_size=0.25, random_state=SEED,
        )
    split_configs.append(("random split (leaky)", train_idx_r, test_idx_r))

    for split_name, tr, te in split_configs:
        X_tr_sp, X_te_sp = X_filt[tr], X_filt[te]
        X_tr_svd, X_te_svd = X_svd_filt[tr], X_svd_filt[te]
        y_tr, y_te = y[tr], y[te]

        if len(set(y_tr)) < 2 or len(y_te) < 1:
            continue

        models = [
            ("Majority baseline", DummyClassifier(strategy="most_frequent", random_state=SEED)),
            ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=SEED, n_jobs=-1)),
            ("LinearSVC", LinearSVC(max_iter=2000, random_state=SEED, dual="auto")),
            ("kNN (cosine)", KNeighborsClassifier(n_neighbors=min(5, len(y_tr)), metric="cosine")),
        ]

        split_results = []
        for mname, model in models:
            # Sparse input for linear models and kNN
            if sparse.issparse(X_tr_sp):
                res = _train_eval_model(model, X_tr_sp, X_te_sp, y_tr, y_te, mname)
            else:
                res = _train_eval_model(model, X_tr_sp, X_te_sp, y_tr, y_te, mname)
            res["majority_baseline_acc"] = majority_acc
            split_results.append(res)

        # RandomForest on SVD features (optional)
        rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
        res_rf = _train_eval_model(rf, X_tr_svd, X_te_svd, y_tr, y_te, "RandomForest (SVD)")
        res_rf["majority_baseline_acc"] = majority_acc
        split_results.append(res_rf)

        results[split_name] = split_results

    return results


# ---------------------------------------------------------------------------
# Abundance
# ---------------------------------------------------------------------------

def load_metadata_weights(
    meta_bytes: Optional[bytes],
    seq_ids: List[str],
    weight_col: str = "weight",
) -> Tuple[np.ndarray, int]:
    """
    Load weights from metadata CSV. Unmatched IDs and missing weights
    default to 1.0. Returns (weights_array, n_matched).
    """
    default = 1.0
    n_matched = 0
    if meta_bytes is None:
        return np.full(len(seq_ids), default), 0
    try:
        dfm = pd.read_csv(io.BytesIO(meta_bytes))
    except Exception:
        return np.full(len(seq_ids), default), 0
    if "id" not in dfm.columns:
        return np.full(len(seq_ids), default), 0

    # Find weight column
    actual_col = None
    if weight_col in dfm.columns:
        actual_col = weight_col
    else:
        for c in ["weight", "abundance", "depth", "reads", "count"]:
            if c in dfm.columns:
                actual_col = c
                break
    if actual_col is None:
        return np.full(len(seq_ids), default), 0

    wmap = dict(zip(
        dfm["id"].astype(str),
        pd.to_numeric(dfm[actual_col], errors="coerce").fillna(default),
    ))
    weights = np.array([wmap.get(str(sid), default) for sid in seq_ids], dtype=float)
    n_matched = sum(1 for sid in seq_ids if str(sid) in wmap)
    return weights, n_matched


def compute_cluster_abundance(
    labels: np.ndarray, weights: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """Compute (weighted) counts per cluster and relative abundance."""
    if labels.size == 0:
        return pd.DataFrame(columns=["cluster", "count", "relative_abundance"])
    if weights is None:
        weights = np.ones_like(labels, dtype=float)
    df = pd.DataFrame({"cluster": labels, "w": weights})
    agg = df.groupby("cluster", as_index=False)["w"].sum().rename(columns={"w": "count"})
    total = agg["count"].sum()
    agg["relative_abundance"] = agg["count"] / total if total > 0 else 0.0
    agg["sort_key"] = agg["cluster"].apply(lambda c: (1, c) if c == -1 else (0, c))
    agg = agg.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)
    return agg


# ---------------------------------------------------------------------------
# Run metadata / reproducibility
# ---------------------------------------------------------------------------

def build_run_metadata(
    k: int,
    max_features: int,
    max_sequences: int,
    per_class_cap: int,
    min_class_size: int,
    label_col: str,
    use_tfidf: bool,
    audit: Dict,
    cluster_metrics: Dict,
    classification_results: Dict,
    k_clusters: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a JSON-serializable run summary."""
    import sklearn, scipy
    from Bio import __version__ as bio_ver

    meta = {
        "seed": SEED,
        "k": k,
        "k_clusters": k_clusters if k_clusters is not None else cluster_metrics.get("k"),
        "max_features": max_features,
        "max_sequences": max_sequences,
        "per_class_cap": per_class_cap,
        "min_class_size": min_class_size,
        "label_col": label_col,
        "use_tfidf": use_tfidf,
        "audit": {k2: v2 for k2, v2 in audit.items() if k2 not in ("genus_counts", "species_counts")},
        "cluster_eval": {k2: v2 for k2, v2 in cluster_metrics.items() if not isinstance(v2, np.ndarray)},
        "library_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "biopython": bio_ver,
        },
    }
    # Add classification accuracy summaries (avoid non-serializable objects)
    if "error" not in classification_results:
        clf_summary = {}
        for split_name, models in classification_results.items():
            if isinstance(models, list):
                clf_summary[split_name] = [
                    {"model": m["name"], "accuracy": m["accuracy"],
                     "balanced_accuracy": m["balanced_accuracy"],
                     "macro_f1": m["macro_f1"]}
                    for m in models
                ]
        meta["classification"] = clf_summary
    return meta
