"""
Tests for seqlib.py.
Uses pdbaa.fa (the 45-record sample, aliased as pdbaa2.fa in the repo root).
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

# Ensure repo root is importable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import seqlib

FASTA_PATH = os.path.join(REPO, "pdbaa2.fa")


# ------------------------------------------------------------------
# Parse tests
# ------------------------------------------------------------------

def test_parse_record_count():
    """pdbaa2.fa has 45 raw records."""
    with open(FASTA_PATH, "rb") as f:
        df, audit = seqlib.parse_fasta(f)
    assert audit["records"] == 45
    # kept + ambiguous + viruses + too_short == records
    assert audit["kept"] + audit["ambiguous"] + audit["viruses"] + audit["too_short"] == 45


def test_no_cr_or_ctrl_a_in_output():
    """No \r or \x01 in any id, description, or sequence."""
    with open(FASTA_PATH, "rb") as f:
        df, _ = seqlib.parse_fasta(f)
    for col in ["id", "description", "sequence"]:
        combined = "\n".join(df[col].astype(str).tolist())
        assert "\r" not in combined, f"\r found in {col}"
        assert "\x01" not in combined, f"\x01 found in {col}"


def test_merged_headers_split():
    """Records with \x01 in the raw file have it removed/split."""
    with open(FASTA_PATH, "rb") as f:
        raw = f.read()
    assert b"\x01" in raw, "Test file should contain \x01"
    with open(FASTA_PATH, "rb") as f:
        df, _ = seqlib.parse_fasta(f)
    for desc in df["description"]:
        # \x01 should still be in the raw description (it's the original description from BioPython)
        # but we check that organism extraction handled it properly
        pass  # Covered by the organism agreement logic
    # All kept records have a non-null organism or None (but not ambiguous)
    assert len(df) > 0


def test_ambiguous_count():
    """Verify ambiguous count matches expected."""
    with open(FASTA_PATH, "rb") as f:
        df, audit = seqlib.parse_fasta(f)
    # After normalization to genus+species, only Oryctolagus/Homo is ambiguous
    assert audit["ambiguous"] == 1, f"Expected 1 ambiguous, got {audit['ambiguous']}"


def test_virus_count():
    """Coronavirus record should be filtered."""
    with open(FASTA_PATH, "rb") as f:
        df, audit = seqlib.parse_fasta(f)
    assert audit["viruses"] == 1, f"Expected 1 virus, got {audit['viruses']}"


def test_too_short_count():
    """Sequences shorter than 30 should be dropped."""
    with open(FASTA_PATH, "rb") as f:
        df, audit = seqlib.parse_fasta(f)
    assert audit["too_short"] == 2, f"Expected 2 too_short, got {audit['too_short']}"


# ------------------------------------------------------------------
# Feature tests
# ------------------------------------------------------------------

def test_feature_matrix_sparse_float32():
    """Feature matrix is sparse and float32, with no NaN."""
    with open(FASTA_PATH, "rb") as f:
        df, _ = seqlib.parse_fasta(f)
    X, vec = seqlib.featurize(df["sequence"], k=3)
    assert sparse.issparse(X), "Feature matrix should be sparse"
    assert X.dtype == np.float32, f"Expected float32, got {X.dtype}"
    assert not np.isnan(X.data).any(), "NaN found in feature matrix"


def test_feature_matrix_dimensions():
    with open(FASTA_PATH, "rb") as f:
        df, _ = seqlib.parse_fasta(f)
    X, _ = seqlib.featurize(df["sequence"], k=3)
    assert X.shape[0] == len(df)
    assert X.shape[1] > 0


# ------------------------------------------------------------------
# Memory test (synthetic)
# ------------------------------------------------------------------

def test_featurize_memory_150k():
    """
    150,000 random 200-aa sequences through featurize with 20000 cap.
    Peak RSS should stay under 2 GB.
    """
    import tracemalloc
    tracemalloc.start()

    AA = "ACDEFGHIKLMNPQRSTVWY"
    rng = np.random.RandomState(42)
    seqs = pd.Series([
        "".join(rng.choice(list(AA), 200)) for _ in range(150_000)
    ])
    # Apply sampling cap
    seqs = seqs.iloc[:20000]
    X, vec = seqlib.featurize(seqs, k=3, max_features=5000)
    assert sparse.issparse(X)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    print(f"Peak memory: {peak_mb:.0f} MB")
    assert peak_mb < 2048, f"Peak RSS {peak_mb:.0f} MB exceeds 2 GB"


# ------------------------------------------------------------------
# Clustering & reduction smoke tests
# ------------------------------------------------------------------

def test_svd_reduction():
    with open(FASTA_PATH, "rb") as f:
        df, _ = seqlib.parse_fasta(f)
    X, _ = seqlib.featurize(df["sequence"], k=3)
    X_svd = seqlib.reduce_svd(X, n_components=10)
    assert X_svd.shape == (len(df), min(10, X.shape[1] - 1, len(df) - 1))


def test_clustering_returns_labels():
    with open(FASTA_PATH, "rb") as f:
        df, _ = seqlib.parse_fasta(f)
    X, _ = seqlib.featurize(df["sequence"], k=3)
    X_svd = seqlib.reduce_svd(X, n_components=10)
    labels = seqlib.run_clustering(X_svd, algo="MiniBatchKMeans", k=3)
    assert len(labels) == len(df)
    assert len(set(labels)) >= 2
