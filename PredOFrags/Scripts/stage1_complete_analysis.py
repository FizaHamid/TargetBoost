#!/usr/bin/env python3

"""
STAGE 1: Statistical and Sequence-Level Characterization of tRF Dataset
=========================================================================

Hierarchy:

smallRNA
   |
   +-- non-tRF
   |
   +-- tRF
        |
        +-- 3tRH
        +-- 5tRH
        +-- tRF-1
        +-- tRF-3
        +-- tRF-5
        +-- leader
        +-- other

This script performs:

1. Dataset audit and FASTA QC
2. Exact duplicate analysis
3. Sequence-level feature extraction
4. Nucleotide composition
5. k-mer feature extraction (1-mer, 2-mer, 3-mer)
6. Length statistics
7. Binary tRF vs non-tRF statistical analysis
8. tRF subtype statistical analysis
9. Pairwise statistical testing
10. Multiple-testing correction
11. Effect sizes
12. KS distribution comparisons
13. k-mer enrichment
14. Feature correlation analysis
15. Mutual information
16. PCA
17. UMAP (if installed)
18. Optional CD-HIT redundancy analysis
19. Publication-ready plotting data tables

The script intentionally does NOT train ML/DL models.
"""

import os
import sys
import re
import json
import math
import time
import hashlib
import shutil
import subprocess
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import (
    mannwhitneyu,
    kruskal,
    ks_2samp,
    chi2_contingency
)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("/lustre/home/sklab202/sourav_ml/sheetal_ml")

POS_DIR = BASE_DIR / "cleaned_dataset/merged_pos_data"
NEG_DIR = BASE_DIR / "cleaned_dataset/negative_dataset"

OUT_DIR = BASE_DIR / "stage1_audit"

POSITIVE_ALL = POS_DIR / "all_positive_merged_unique.fa"
NEGATIVE_ALL = NEG_DIR / "all_negative.fa"

SUBTYPE_FILES = {
    "3tRH": POS_DIR / "3tRH.fa",
    "5tRH": POS_DIR / "5tRH.fa",
    "tRF-1": POS_DIR / "tRF_1.fa",
    "tRF-3": POS_DIR / "tRF_3_all.fa",
    "tRF-5": POS_DIR / "tRF_5.fa",
    "leader": POS_DIR / "leader_tRF.fa",
    "other": POS_DIR / "other_tRF.fa",
}

# Maximum number of sequences used for UMAP.
UMAP_SAMPLE_SIZE = 20000

# Maximum sequences used for PCA fitting if memory is limiting.
PCA_SAMPLE_SIZE = 100000

RANDOM_SEED = 42

# k-mer sizes
KMER_SIZES = [1, 2, 3]

# Characters allowed in canonical RNA sequences
VALID_BASES = set("ACGU")

# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

DIR_AUDIT = OUT_DIR / "01_dataset_audit"
DIR_FEATURES = OUT_DIR / "02_sequence_features"
DIR_STATS = OUT_DIR / "03_statistical_analysis"
DIR_REDUNDANCY = OUT_DIR / "04_sequence_redundancy"
DIR_VIS = OUT_DIR / "05_visualization_data"
DIR_LOG = OUT_DIR / "06_logs"

for d in [
    DIR_AUDIT,
    DIR_FEATURES,
    DIR_STATS,
    DIR_REDUNDANCY,
    DIR_VIS,
    DIR_LOG,
]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def log(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)

    with open(DIR_LOG / "stage1.log", "a") as f:
        f.write(line + "\n")


def safe_filename(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def bh_fdr(pvalues):
    """
    Benjamini-Hochberg FDR correction.
    """
    pvalues = np.asarray(pvalues, dtype=float)

    n = len(pvalues)

    if n == 0:
        return np.array([])

    order = np.argsort(pvalues)
    ranked = pvalues[order]

    q = ranked * n / np.arange(1, n + 1)

    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    result = np.empty(n)
    result[order] = q

    return result


def cliffs_delta_from_u(u, n1, n2):
    """
    Cliff's delta derived from Mann-Whitney U.

    Range:
        -1 = group 1 lower than group 2
         0 = identical distributions
        +1 = group 1 higher than group 2
    """
    if n1 == 0 or n2 == 0:
        return np.nan

    return (2.0 * u / (n1 * n2)) - 1.0


def interpretation_cliffs(delta):
    if pd.isna(delta):
        return "NA"

    a = abs(delta)

    if a < 0.147:
        return "negligible"
    elif a < 0.33:
        return "small"
    elif a < 0.474:
        return "medium"
    else:
        return "large"


# ============================================================
# FASTA READER
# ============================================================

def canonicalize_sequence(seq):
    """
    Convert DNA-style T to RNA-style U.
    Remove whitespace and convert uppercase.
    """
    seq = re.sub(r"\s+", "", seq.upper())
    seq = seq.replace("T", "U")
    return seq


def read_fasta(path):
    """
    Robust FASTA parser.

    Returns:
        list of (header, sequence)
    """

    records = []

    header = None
    seq_parts = []

    with open(path, "r", errors="replace") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):

                if header is not None:
                    seq = canonicalize_sequence("".join(seq_parts))
                    records.append((header, seq))

                header = line[1:]
                seq_parts = []

            else:
                seq_parts.append(line)

        if header is not None:
            seq = canonicalize_sequence("".join(seq_parts))
            records.append((header, seq))

    return records


# ============================================================
# DATASET DEFINITIONS
# ============================================================

def load_dataset(path, label, subclass):

    log(f"Reading: {path}")

    records = read_fasta(path)

    rows = []

    for i, (header, sequence) in enumerate(records):

        seq_id = header.split()[0] if header else f"seq_{i+1}"

        rows.append({
            "sequence_id": seq_id,
            "header": header,
            "sequence": sequence,
            "label": label,
            "label_numeric": 1 if label == "tRF" else 0,
            "subclass_source": subclass,
            "source_file": path.name,
        })

    return rows


# ============================================================
# STEP 1 — DATASET AUDIT
# ============================================================

def dataset_audit():

    log("=" * 70)
    log("STEP 1: DATASET AUDIT")
    log("=" * 70)

    all_rows = []

    # Positive
    positive_rows = load_dataset(
        POSITIVE_ALL,
        "tRF",
        "all_tRF"
    )

    all_rows.extend(positive_rows)

    # Negative
    negative_rows = load_dataset(
        NEGATIVE_ALL,
        "non-tRF",
        "non_tRF"
    )

    all_rows.extend(negative_rows)

    # subtype membership
    subtype_membership = {}

    subtype_counts = []

    for subtype, path in SUBTYPE_FILES.items():

        rows = read_fasta(path)

        seqs = set()

        for header, seq in rows:
            seqs.add(seq)

        subtype_membership[subtype] = seqs

        subtype_counts.append({
            "subclass": subtype,
            "records_in_fasta": len(rows),
            "unique_sequences": len(seqs),
            "file": path.name,
        })

    subtype_count_df = pd.DataFrame(subtype_counts)

    subtype_count_df.to_csv(
        DIR_AUDIT / "subtype_file_counts.tsv",
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # Determine subtype membership
    # --------------------------------------------------------

    membership_rows = []

    for seq in set(
        r["sequence"]
        for r in positive_rows
    ):

        members = [
            subtype
            for subtype, seqset in subtype_membership.items()
            if seq in seqset
        ]

        membership_rows.append({
            "sequence": seq,
            "n_subtypes": len(members),
            "subtypes": ";".join(members) if members else "UNASSIGNED",
            "subclass_assignment": (
                members[0]
                if len(members) == 1
                else (
                    "MULTIPLE"
                    if len(members) > 1
                    else "UNASSIGNED"
                )
            )
        })

    membership_df = pd.DataFrame(membership_rows)

    membership_df.to_csv(
        DIR_AUDIT / "positive_subtype_membership.tsv",
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # Attach subtype to main data
    # --------------------------------------------------------

    membership_map = dict(
        zip(
            membership_df["sequence"],
            membership_df["subclass_assignment"]
        )
    )

    for row in all_rows:

        if row["label"] == "tRF":
            row["subclass"] = membership_map.get(
                row["sequence"],
                "UNASSIGNED"
            )
        else:
            row["subclass"] = "non_tRF"

    df = pd.DataFrame(all_rows)

    # --------------------------------------------------------
    # Basic QC
    # --------------------------------------------------------

    qc_rows = []

    for source_file, group in df.groupby("source_file"):

        sequences = group["sequence"]

        invalid_mask = ~sequences.apply(
            lambda x: set(x).issubset(VALID_BASES)
        )

        empty_mask = sequences.str.len() == 0

        duplicate_count = len(sequences) - sequences.nunique()

        qc_rows.append({

            "source_file": source_file,
            "records": len(group),
            "unique_sequences": sequences.nunique(),
            "duplicate_records": duplicate_count,
            "duplicate_percent": (
                duplicate_count / len(group) * 100
                if len(group) > 0 else np.nan
            ),
            "empty_sequences": empty_mask.sum(),
            "invalid_sequences": invalid_mask.sum(),
            "min_length": sequences.str.len().min(),
            "median_length": sequences.str.len().median(),
            "mean_length": sequences.str.len().mean(),
            "std_length": sequences.str.len().std(),
            "max_length": sequences.str.len().max(),
        })

    qc_df = pd.DataFrame(qc_rows)

    qc_df.to_csv(
        DIR_AUDIT / "dataset_qc_summary.tsv",
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # Class summary
    # --------------------------------------------------------

    class_summary = []

    for label, group in df.groupby("label"):

        class_summary.append({

            "label": label,
            "n_records": len(group),
            "n_unique_sequences": group["sequence"].nunique(),
            "duplicate_records": (
                len(group) - group["sequence"].nunique()
            ),
            "percentage_dataset": (
                len(group) / len(df) * 100
            ),
            "min_length": group["sequence"].str.len().min(),
            "q1_length": group["sequence"].str.len().quantile(0.25),
            "median_length": group["sequence"].str.len().median(),
            "mean_length": group["sequence"].str.len().mean(),
            "q3_length": group["sequence"].str.len().quantile(0.75),
            "max_length": group["sequence"].str.len().max(),
            "std_length": group["sequence"].str.len().std(),
        })

    class_summary_df = pd.DataFrame(class_summary)

    class_summary_df.to_csv(
        DIR_AUDIT / "class_summary.tsv",
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # Subclass summary
    # --------------------------------------------------------

    subclass_summary = []

    for subclass, group in df[df["label"] == "tRF"].groupby("subclass"):

        subclass_summary.append({

            "subclass": subclass,
            "n_records": len(group),
            "n_unique_sequences": group["sequence"].nunique(),
            "percentage_of_tRF": (
                len(group) /
                len(df[df["label"] == "tRF"]) *
                100
            ),
            "min_length": group["sequence"].str.len().min(),
            "q1_length": group["sequence"].str.len().quantile(0.25),
            "median_length": group["sequence"].str.len().median(),
            "mean_length": group["sequence"].str.len().mean(),
            "q3_length": group["sequence"].str.len().quantile(0.75),
            "max_length": group["sequence"].str.len().max(),
            "std_length": group["sequence"].str.len().std(),
        })

    subclass_summary_df = pd.DataFrame(subclass_summary)

    subclass_summary_df.to_csv(
        DIR_AUDIT / "subclass_summary.tsv",
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # Save master audit metadata
    # --------------------------------------------------------

    df.to_csv(
        DIR_AUDIT / "master_sequence_metadata.tsv",
        sep="\t",
        index=False
    )

    log(f"Total records: {len(df):,}")
    log(f"Unique sequences: {df.sequence.nunique():,}")
    log(f"tRF records: {(df.label == 'tRF').sum():,}")
    log(f"non-tRF records: {(df.label == 'non-tRF').sum():,}")

    return df, subtype_membership


# ============================================================
# STEP 2 — EXACT DUPLICATE ANALYSIS
# ============================================================

def duplicate_analysis(df):

    log("=" * 70)
    log("STEP 2: DUPLICATE ANALYSIS")
    log("=" * 70)

    rows = []

    for label, group in df.groupby("label"):

        counts = group["sequence"].value_counts()

        rows.append({

            "label": label,
            "records": len(group),
            "unique_sequences": len(counts),
            "duplicate_records": len(group) - len(counts),
            "duplicate_fraction": (
                (len(group) - len(counts)) / len(group)
            ),
            "max_duplicate_count": counts.max(),
            "sequences_occurring_once": (counts == 1).sum(),
            "sequences_occurring_multiple_times": (counts > 1).sum(),
        })

    summary = pd.DataFrame(rows)

    summary.to_csv(
        DIR_REDUNDANCY / "exact_duplicate_summary.tsv",
        sep="\t",
        index=False
    )

    # Sequence-level duplicate distribution
    duplicate_rows = []

    for seq, count in df["sequence"].value_counts().items():

        if count > 1:

            duplicate_rows.append({
                "sequence_hash": hashlib.md5(
                    seq.encode()
                ).hexdigest(),
                "sequence_length": len(seq),
                "occurrence_count": count,
            })

    duplicate_df = pd.DataFrame(duplicate_rows)

    duplicate_df.to_csv(
        DIR_REDUNDANCY / "duplicate_sequence_distribution.tsv",
        sep="\t",
        index=False
    )

    # Cross-class identical sequences
    pos_set = set(
        df.loc[df["label"] == "tRF", "sequence"]
    )

    neg_set = set(
        df.loc[df["label"] == "non-tRF", "sequence"]
    )

    overlap = pos_set.intersection(neg_set)

    cross_class = pd.DataFrame({

        "metric": [
            "unique_tRF",
            "unique_non_tRF",
            "identical_sequences_shared_between_classes"
        ],

        "value": [
            len(pos_set),
            len(neg_set),
            len(overlap)
        ]
    })

    cross_class.to_csv(
        DIR_REDUNDANCY / "cross_class_exact_overlap.tsv",
        sep="\t",
        index=False
    )

    if overlap:

        overlap_df = pd.DataFrame({
            "sequence": list(overlap),
            "length": [
                len(x) for x in overlap
            ]
        })

        overlap_df.to_csv(
            DIR_REDUNDANCY /
            "cross_class_exact_overlap_sequences.tsv",
            sep="\t",
            index=False
        )

    log(
        f"Cross-class exact sequence overlap: "
        f"{len(overlap):,}"
    )


# ============================================================
# STEP 3 — SEQUENCE FEATURES
# ============================================================

def calculate_entropy(a, c, g, u):

    counts = np.array([a, c, g, u], dtype=float)

    total = counts.sum()

    if total == 0:
        return 0.0

    p = counts / total

    p = p[p > 0]

    return float(
        -np.sum(p * np.log2(p))
    )


def calculate_kmer_counts(sequence, k):

    if len(sequence) < k:
        return {}

    counts = {}

    for i in range(len(sequence) - k + 1):

        kmer = sequence[i:i+k]

        counts[kmer] = counts.get(kmer, 0) + 1

    return counts


def extract_features(df):

    log("=" * 70)
    log("STEP 3: SEQUENCE FEATURE EXTRACTION")
    log("=" * 70)

    bases = ["A", "C", "G", "U"]

    all_kmers = {}

    for k in KMER_SIZES:

        all_kmers[k] = [
            "".join(x)
            for x in product(bases, repeat=k)
        ]

    feature_rows = []

    for idx, row in enumerate(
        df.itertuples(index=False),
        start=1
    ):

        seq = row.sequence
        L = len(seq)

        counts = {
            base: seq.count(base)
            for base in bases
        }

        a = counts["A"]
        c = counts["C"]
        g = counts["G"]
        u = counts["U"]

        total = max(L, 1)

        gc = (g + c) / total
        au = (a + u) / total

        gc_skew = (
            (g - c) / (g + c)
            if (g + c) > 0
            else 0
        )

        au_skew = (
            (a - u) / (a + u)
            if (a + u) > 0
            else 0
        )

        purine = (a + g) / total
        pyrimidine = (c + u) / total

        row_features = {

            "sequence_id": row.sequence_id,
            "sequence": seq,
            "label": row.label,
            "label_numeric": row.label_numeric,
            "subclass": row.subclass,
            "source_file": row.source_file,

            "length": L,

            "A_count": a,
            "C_count": c,
            "G_count": g,
            "U_count": u,

            "A_fraction": a / total,
            "C_fraction": c / total,
            "G_fraction": g / total,
            "U_fraction": u / total,

            "GC_fraction": gc,
            "AU_fraction": au,

            "GC_skew": gc_skew,
            "AU_skew": au_skew,

            "purine_fraction": purine,
            "pyrimidine_fraction": pyrimidine,

            "shannon_entropy": calculate_entropy(
                a, c, g, u
            ),
        }

        # k-mer normalized counts
        for k in KMER_SIZES:

            counts_k = calculate_kmer_counts(seq, k)

            denominator = max(
                L - k + 1,
                1
            )

            for kmer in all_kmers[k]:

                row_features[
                    f"kmer_{k}_{kmer}"
                ] = (
                    counts_k.get(kmer, 0)
                    / denominator
                )

        feature_rows.append(row_features)

        if idx % 25000 == 0:
            log(
                f"Feature extraction: "
                f"{idx:,}/{len(df):,}"
            )

    feature_df = pd.DataFrame(feature_rows)

    feature_df.to_csv(
        DIR_FEATURES /
        "sequence_features.tsv",
        sep="\t",
        index=False
    )

    # separate feature matrix without sequence
    numeric_cols = [
        c for c in feature_df.columns
        if c not in [
            "sequence_id",
            "sequence",
            "label",
            "subclass",
            "source_file"
        ]
    ]

    feature_matrix = feature_df[
        numeric_cols
    ]

    feature_matrix.to_csv(
        DIR_FEATURES /
        "numeric_feature_matrix.tsv",
        sep="\t",
        index=False
    )

    # nucleotide composition
    composition_cols = [
        "sequence_id",
        "label",
        "subclass",
        "length",
        "A_fraction",
        "C_fraction",
        "G_fraction",
        "U_fraction",
        "GC_fraction",
        "AU_fraction",
        "GC_skew",
        "AU_skew",
        "shannon_entropy"
    ]

    feature_df[
        composition_cols
    ].to_csv(
        DIR_FEATURES /
        "nucleotide_composition.tsv",
        sep="\t",
        index=False
    )

    # k-mer-only matrix
    kmer_cols = [
        c for c in feature_df.columns
        if c.startswith("kmer_")
    ]

    feature_df[
        [
            "sequence_id",
            "label",
            "subclass"
        ] + kmer_cols
    ].to_csv(
        DIR_FEATURES /
        "kmer_features.tsv",
        sep="\t",
        index=False
    )

    log(
        f"Feature matrix generated: "
        f"{feature_df.shape[0]:,} sequences × "
        f"{feature_df.shape[1]:,} columns"
    )

    return feature_df


# ============================================================
# STEP 4 — DESCRIPTIVE STATISTICS
# ============================================================

def descriptive_statistics(feature_df):

    log("=" * 70)
    log("STEP 4: DESCRIPTIVE STATISTICS")
    log("=" * 70)

    continuous_features = [
        "length",
        "A_fraction",
        "C_fraction",
        "G_fraction",
        "U_fraction",
        "GC_fraction",
        "AU_fraction",
        "GC_skew",
        "AU_skew",
        "purine_fraction",
        "pyrimidine_fraction",
        "shannon_entropy",
    ]

    rows = []

    for feature in continuous_features:

        for group_col in ["label", "subclass"]:

            for group, g in feature_df.groupby(
                group_col,
                dropna=False
            ):

                values = g[feature].dropna()

                if len(values) == 0:
                    continue

                rows.append({

                    "feature": feature,
                    "group_type": group_col,
                    "group": group,

                    "n": len(values),

                    "mean": values.mean(),
                    "std": values.std(),

                    "min": values.min(),
                    "q1": values.quantile(0.25),
                    "median": values.median(),
                    "q3": values.quantile(0.75),
                    "max": values.max(),

                    "IQR": (
                        values.quantile(0.75)
                        -
                        values.quantile(0.25)
                    ),

                    "MAD": (
                        np.median(
                            np.abs(
                                values -
                                np.median(values)
                            )
                        )
                    ),

                    "skewness": values.skew(),
                    "kurtosis": values.kurtosis(),
                })

    descriptive_df = pd.DataFrame(rows)

    descriptive_df.to_csv(
        DIR_STATS /
        "descriptive_statistics.tsv",
        sep="\t",
        index=False
    )

    # Length histogram bins
    length_bins = (
        feature_df[
            ["label", "subclass", "length"]
        ]
        .copy()
    )

    length_bins.to_csv(
        DIR_VIS /
        "length_distribution_data.tsv",
        sep="\t",
        index=False
    )

    # Composition plotting data
    composition_plot = feature_df[
        [
            "sequence_id",
            "label",
            "subclass",
            "A_fraction",
            "C_fraction",
            "G_fraction",
            "U_fraction",
            "GC_fraction",
            "AU_fraction",
            "shannon_entropy",
        ]
    ]

    composition_plot.to_csv(
        DIR_VIS /
        "composition_plot_data.tsv",
        sep="\t",
        index=False
    )


# ============================================================
# STEP 5 — BINARY STATISTICS
# ============================================================

def binary_statistics(feature_df):

    log("=" * 70)
    log("STEP 5: tRF vs non-tRF STATISTICS")
    log("=" * 70)

    features = [
        "length",
        "A_fraction",
        "C_fraction",
        "G_fraction",
        "U_fraction",
        "GC_fraction",
        "AU_fraction",
        "GC_skew",
        "AU_skew",
        "purine_fraction",
        "pyrimidine_fraction",
        "shannon_entropy",
    ]

    rows = []

    tRF = feature_df[
        feature_df["label"] == "tRF"
    ]

    non_tRF = feature_df[
        feature_df["label"] == "non-tRF"
    ]

    for feature in features:

        x = tRF[feature].dropna().values
        y = non_tRF[feature].dropna().values

        U, p = mannwhitneyu(
            x,
            y,
            alternative="two-sided"
        )

        delta = cliffs_delta_from_u(
            U,
            len(x),
            len(y)
        )

        ks_D, ks_p = ks_2samp(
            x,
            y
        )

        rows.append({

            "feature": feature,

            "n_tRF": len(x),
            "n_non_tRF": len(y),

            "tRF_median": np.median(x),
            "non_tRF_median": np.median(y),

            "tRF_mean": np.mean(x),
            "non_tRF_mean": np.mean(y),

            "mann_whitney_U": U,
            "mann_whitney_p": p,

            "cliffs_delta": delta,
            "cliffs_delta_abs": abs(delta),
            "effect_interpretation":
                interpretation_cliffs(delta),

            "ks_D": ks_D,
            "ks_p": ks_p,
        })

    result = pd.DataFrame(rows)

    result["mann_whitney_FDR"] = bh_fdr(
        result["mann_whitney_p"]
    )

    result["ks_FDR"] = bh_fdr(
        result["ks_p"]
    )

    result.to_csv(
        DIR_STATS /
        "binary_tRF_vs_non_tRF_statistics.tsv",
        sep="\t",
        index=False
    )


# ============================================================
# STEP 6 — SUBCLASS STATISTICS
# ============================================================

def subclass_statistics(feature_df):

    log("=" * 70)
    log("STEP 6: SUBCLASS STATISTICS")
    log("=" * 70)

    tRF_df = feature_df[
        feature_df["label"] == "tRF"
    ].copy()

    features = [
        "length",
        "A_fraction",
        "C_fraction",
        "G_fraction",
        "U_fraction",
        "GC_fraction",
        "AU_fraction",
        "GC_skew",
        "AU_skew",
        "purine_fraction",
        "pyrimidine_fraction",
        "shannon_entropy",
    ]

    classes = sorted(
        tRF_df["subclass"].unique()
    )

    # --------------------------------------------------------
    # Kruskal-Wallis
    # --------------------------------------------------------

    kw_rows = []

    for feature in features:

        groups = [
            tRF_df[
                tRF_df["subclass"] == cls
            ][feature].dropna().values
            for cls in classes
        ]

        valid_groups = [
            x for x in groups
            if len(x) > 0
        ]

        if len(valid_groups) < 2:
            continue

        H, p = kruskal(
            *valid_groups
        )

        kw_rows.append({

            "feature": feature,
            "n_classes": len(valid_groups),
            "kruskal_H": H,
            "kruskal_p": p,
        })

    kw_df = pd.DataFrame(kw_rows)

    kw_df["kruskal_FDR"] = bh_fdr(
        kw_df["kruskal_p"]
    )

    kw_df.to_csv(
        DIR_STATS /
        "subclass_kruskal_wallis.tsv",
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # Pairwise Mann-Whitney
    # --------------------------------------------------------

    pair_rows = []

    for feature in features:

        for i in range(len(classes)):

            for j in range(i + 1, len(classes)):

                c1 = classes[i]
                c2 = classes[j]

                x = tRF_df[
                    tRF_df["subclass"] == c1
                ][feature].dropna().values

                y = tRF_df[
                    tRF_df["subclass"] == c2
                ][feature].dropna().values

                if len(x) == 0 or len(y) == 0:
                    continue

                U, p = mannwhitneyu(
                    x,
                    y,
                    alternative="two-sided"
                )

                delta = cliffs_delta_from_u(
                    U,
                    len(x),
                    len(y)
                )

                pair_rows.append({

                    "feature": feature,
                    "class_1": c1,
                    "class_2": c2,

                    "n_class_1": len(x),
                    "n_class_2": len(y),

                    "median_class_1": np.median(x),
                    "median_class_2": np.median(y),

                    "U": U,
                    "p_value": p,

                    "cliffs_delta": delta,
                    "abs_cliffs_delta": abs(delta),

                    "effect_interpretation":
                        interpretation_cliffs(delta),
                })

    pairwise_df = pd.DataFrame(pair_rows)

    if len(pairwise_df) > 0:

        pairwise_df["FDR"] = bh_fdr(
            pairwise_df["p_value"]
        )

    pairwise_df.to_csv(
        DIR_STATS /
        "subclass_pairwise_statistics.tsv",
        sep="\t",
        index=False
    )


# ============================================================
# STEP 7 — K-MER ENRICHMENT
# ============================================================

def kmer_enrichment(feature_df):

    log("=" * 70)
    log("STEP 7: k-mer ENRICHMENT")
    log("=" * 70)

    results = []

    tRF_mask = (
        feature_df["label"] == "tRF"
    )

    non_mask = (
        feature_df["label"] == "non-tRF"
    )

    for k in KMER_SIZES:

        kmers = [
            "".join(x)
            for x in product(
                ["A", "C", "G", "U"],
                repeat=k
            )
        ]

        for kmer in kmers:

            # sequence-level presence
            presence = (
                feature_df["sequence"]
                .str.contains(
                    kmer,
                    regex=False
                )
            )

            a = int(
                (presence & tRF_mask).sum()
            )

            b = int(
                (tRF_mask & ~presence).sum()
            )

            c = int(
                (presence & non_mask).sum()
            )

            d = int(
                (non_mask & ~presence).sum()
            )

            # Haldane correction
            aa = a + 0.5
            bb = b + 0.5
            cc = c + 0.5
            dd = d + 0.5

            odds_ratio = (
                (aa * dd) /
                (bb * cc)
            )

            log2_or = np.log2(
                odds_ratio
            )

            table = np.array([
                [a, b],
                [c, d]
            ])

            chi2, p, _, _ = chi2_contingency(
                table,
                correction=False
            )

            results.append({

                "k": k,
                "kmer": kmer,

                "tRF_present": a,
                "tRF_absent": b,

                "non_tRF_present": c,
                "non_tRF_absent": d,

                "tRF_frequency": (
                    a / (a + b)
                    if (a + b) else np.nan
                ),

                "non_tRF_frequency": (
                    c / (c + d)
                    if (c + d) else np.nan
                ),

                "odds_ratio": odds_ratio,
                "log2_odds_ratio": log2_or,

                "chi2": chi2,
                "p_value": p,
            })

    result = pd.DataFrame(results)

    result["FDR"] = bh_fdr(
        result["p_value"]
    )

    result["direction"] = np.where(
        result["log2_odds_ratio"] > 0,
        "tRF_enriched",
        "non_tRF_enriched"
    )

    result = result.sort_values(
        ["k", "FDR", "log2_odds_ratio"],
        ascending=[True, True, False]
    )

    result.to_csv(
        DIR_STATS /
        "kmer_enrichment_tRF_vs_non_tRF.tsv",
        sep="\t",
        index=False
    )

    # Separate k-mer plot tables
    for k in KMER_SIZES:

        subset = result[
            result["k"] == k
        ].copy()

        subset.to_csv(
            DIR_VIS /
            f"kmer_enrichment_k{k}.tsv",
            sep="\t",
            index=False
        )


# ============================================================
# STEP 8 — FEATURE CORRELATION
# ============================================================

def feature_correlation(feature_df):

    log("=" * 70)
    log("STEP 8: FEATURE CORRELATION")
    log("=" * 70)

    base_features = [
        "length",
        "A_fraction",
        "C_fraction",
        "G_fraction",
        "U_fraction",
        "GC_fraction",
        "AU_fraction",
        "GC_skew",
        "AU_skew",
        "purine_fraction",
        "pyrimidine_fraction",
        "shannon_entropy",
    ]

    X = feature_df[
        base_features
    ].copy()

    pearson = X.corr(
        method="pearson"
    )

    spearman = X.corr(
        method="spearman"
    )

    pearson.to_csv(
        DIR_STATS /
        "feature_correlation_pearson.tsv",
        sep="\t"
    )

    spearman.to_csv(
        DIR_STATS /
        "feature_correlation_spearman.tsv",
        sep="\t"
    )

    # Long format
    rows = []

    for f1 in base_features:

        for f2 in base_features:

            rows.append({

                "feature_1": f1,
                "feature_2": f2,
                "pearson_r": pearson.loc[f1, f2],
                "spearman_rho": spearman.loc[f1, f2],
            })

    pd.DataFrame(rows).to_csv(
        DIR_STATS /
        "feature_correlation_long.tsv",
        sep="\t",
        index=False
    )


# ============================================================
# STEP 9 — MUTUAL INFORMATION
# ============================================================

def mutual_information_analysis(feature_df):

    log("=" * 70)
    log("STEP 9: MUTUAL INFORMATION")
    log("=" * 70)

    # Use a reproducible sample for computational efficiency.
    if len(feature_df) > 100000:

        sample = feature_df.sample(
            n=100000,
            random_state=RANDOM_SEED
        )

    else:

        sample = feature_df.copy()

    base_features = [
        "length",
        "A_fraction",
        "C_fraction",
        "G_fraction",
        "U_fraction",
        "GC_fraction",
        "AU_fraction",
        "GC_skew",
        "AU_skew",
        "purine_fraction",
        "pyrimidine_fraction",
        "shannon_entropy",
    ]

    X = sample[
        base_features
    ].fillna(0)

    y = sample[
        "label_numeric"
    ].values

    mi = mutual_info_classif(
        X,
        y,
        random_state=RANDOM_SEED
    )

    mi_df = pd.DataFrame({

        "feature": base_features,
        "mutual_information": mi
    })

    mi_df = mi_df.sort_values(
        "mutual_information",
        ascending=False
    )

    mi_df.to_csv(
        DIR_STATS /
        "mutual_information_binary.tsv",
        sep="\t",
        index=False
    )


# ============================================================
# STEP 10 — PCA
# ============================================================

def pca_analysis(feature_df):

    log("=" * 70)
    log("STEP 10: PCA")
    log("=" * 70)

    # Use composition + k-mer features.
    exclude = [
        "sequence_id",
        "sequence",
        "label",
        "subclass",
        "source_file",
    ]

    feature_cols = [
        c for c in feature_df.columns
        if c not in exclude
    ]

    X = feature_df[
        feature_cols
    ].replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    # PCA on a reproducible subset for model fitting
    n_fit = min(
        PCA_SAMPLE_SIZE,
        len(X)
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    fit_idx = rng.choice(
        len(X),
        size=n_fit,
        replace=False
    )

    X_fit = X.iloc[
        fit_idx
    ].values

    scaler = StandardScaler()

    X_fit_scaled = scaler.fit_transform(
        X_fit
    )

    n_components = min(
        20,
        X_fit_scaled.shape[1]
    )

    pca = PCA(
        n_components=n_components,
        random_state=RANDOM_SEED
    )

    pca.fit(
        X_fit_scaled
    )

    # Transform complete dataset in chunks
    X_scaled = scaler.transform(
        X.values
    )

    coordinates = pca.transform(
        X_scaled
    )

    pc_cols = [
        f"PC{i+1}"
        for i in range(
            coordinates.shape[1]
        )
    ]

    pca_df = pd.DataFrame(
        coordinates,
        columns=pc_cols
    )

    pca_df.insert(
        0,
        "sequence_id",
        feature_df["sequence_id"].values
    )

    pca_df["label"] = (
        feature_df["label"].values
    )

    pca_df["subclass"] = (
        feature_df["subclass"].values
    )

    pca_df.to_csv(
        DIR_VIS /
        "PCA_coordinates.tsv",
        sep="\t",
        index=False
    )

    explained = pd.DataFrame({

        "component": pc_cols,

        "explained_variance_ratio":
            pca.explained_variance_ratio_,

        "cumulative_variance":
            np.cumsum(
                pca.explained_variance_ratio_
            ),
    })

    explained.to_csv(
        DIR_STATS /
        "PCA_explained_variance.tsv",
        sep="\t",
        index=False
    )

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=pc_cols
    )

    loadings.to_csv(
        DIR_STATS /
        "PCA_feature_loadings.tsv",
        sep="\t"
    )

    log(
        "PCA completed."
    )


# ============================================================
# STEP 11 — UMAP
# ============================================================

def umap_analysis(feature_df):

    log("=" * 70)
    log("STEP 11: UMAP")
    log("=" * 70)

    try:

        import umap

    except ImportError:

        log(
            "umap-learn is not installed. "
            "UMAP skipped."
        )

        return

    n = min(
        UMAP_SAMPLE_SIZE,
        len(feature_df)
    )

    sample = feature_df.sample(
        n=n,
        random_state=RANDOM_SEED
    ).copy()

    exclude = [
        "sequence_id",
        "sequence",
        "label",
        "subclass",
        "source_file",
    ]

    feature_cols = [
        c for c in sample.columns
        if c not in exclude
    ]

    X = sample[
        feature_cols
    ].replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.3,
        metric="euclidean",
        random_state=RANDOM_SEED,
        n_components=2,
    )

    embedding = reducer.fit_transform(
        X_scaled
    )

    umap_df = pd.DataFrame({

        "sequence_id":
            sample["sequence_id"].values,

        "label":
            sample["label"].values,

        "subclass":
            sample["subclass"].values,

        "UMAP1":
            embedding[:, 0],

        "UMAP2":
            embedding[:, 1],
    })

    umap_df.to_csv(
        DIR_VIS /
        "UMAP_coordinates.tsv",
        sep="\t",
        index=False
    )

    log(
        f"UMAP generated for {n:,} sequences."
    )


# ============================================================
# STEP 12 — OPTIONAL CD-HIT REDUNDANCY ANALYSIS
# ============================================================

def cd_hit_analysis():

    log("=" * 70)
    log("STEP 12: CD-HIT REDUNDANCY ANALYSIS")
    log("=" * 70)

    cd_hit = shutil.which("cd-hit")

    if cd_hit is None:

        log(
            "CD-HIT not found in PATH. "
            "Skipping sequence-identity clustering."
        )

        with open(
            DIR_REDUNDANCY /
            "cdhit_status.txt",
            "w"
        ) as f:

            f.write(
                "CD-HIT executable not found.\n"
                "Install/load CD-HIT and rerun Stage 1 "
                "for sequence identity clustering.\n"
            )

        return

    # Reconstruct combined FASTA
    combined_fasta = (
        DIR_REDUNDANCY /
        "combined_tRF_non_tRF.fa"
    )

    with open(
        combined_fasta,
        "w"
    ) as out:

        for fasta in [
            POSITIVE_ALL,
            NEGATIVE_ALL
        ]:

            records = read_fasta(
                fasta
            )

            for i, (header, seq) in enumerate(
                records
            ):

                prefix = (
                    "tRF"
                    if fasta == POSITIVE_ALL
                    else "non_tRF"
                )

                out.write(
                    f">{prefix}_{i}\n"
                )

                out.write(
                    f"{seq}\n"
                )

    thresholds = [
        1.00,
        0.99,
        0.95,
        0.90,
        0.80,
    ]

    summary = []

    for identity in thresholds:

        tag = str(identity).replace(
            ".",
            ""
        )

        output_prefix = (
            DIR_REDUNDANCY /
            f"cdhit_{tag}"
        )

        cmd = [
            cd_hit,

            "-i",
            str(combined_fasta),

            "-o",
            str(output_prefix),

            "-c",
            str(identity),

            "-n",
            "5" if identity >= 0.9 else "4",

            "-T",
            "0",

            "-M",
            "0",

            "-d",
            "0",
        ]

        log(
            "Running CD-HIT at "
            f"{identity:.2f} identity"
        )

        try:

            subprocess.run(
                cmd,
                check=True
            )

        except Exception as e:

            log(
                f"CD-HIT failed at "
                f"{identity}: {e}"
            )

            continue

        clstr = Path(
            str(output_prefix) +
            ".clstr"
        )

        if not clstr.exists():
            continue

        cluster_count = 0
        sequence_count = 0

        cluster_sizes = []

        current_size = 0

        with open(clstr) as f:

            for line in f:

                line = line.strip()

                if line.startswith(">Cluster"):

                    if current_size > 0:
                        cluster_sizes.append(
                            current_size
                        )

                    cluster_count += 1
                    current_size = 0

                elif line:

                    sequence_count += 1
                    current_size += 1

            if current_size > 0:
                cluster_sizes.append(
                    current_size
                )

        summary.append({

            "identity_threshold": identity,

            "clusters": cluster_count,

            "sequences":
                sequence_count,

            "mean_cluster_size":
                np.mean(cluster_sizes)
                if cluster_sizes
                else np.nan,

            "median_cluster_size":
                np.median(cluster_sizes)
                if cluster_sizes
                else np.nan,

            "max_cluster_size":
                max(cluster_sizes)
                if cluster_sizes
                else np.nan,
        })

    if summary:

        pd.DataFrame(summary).to_csv(
            DIR_REDUNDANCY /
            "cdhit_cluster_summary.tsv",
            sep="\t",
            index=False
        )


# ============================================================
# STEP 13 — MASTER SUMMARY
# ============================================================

def generate_summary(df):

    log("=" * 70)
    log("STEP 13: GENERATING MASTER SUMMARY")
    log("=" * 70)

    summary_file = (
        OUT_DIR /
        "stage1_summary.txt"
    )

    with open(
        summary_file,
        "w"
    ) as f:

        f.write(
            "STAGE 1 — tRF / non-tRF STATISTICAL "
            "CHARACTERIZATION\n"
        )

        f.write("=" * 70 + "\n\n")

        f.write(
            f"Analysis date: "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

        f.write(
            f"Base directory: {BASE_DIR}\n"
        )

        f.write(
            f"Output directory: {OUT_DIR}\n\n"
        )

        f.write(
            "DATASET SUMMARY\n"
        )

        f.write("-" * 50 + "\n")

        f.write(
            f"Total records: {len(df):,}\n"
        )

        f.write(
            f"Unique sequences: "
            f"{df.sequence.nunique():,}\n"
        )

        f.write(
            f"tRF records: "
            f"{(df.label == 'tRF').sum():,}\n"
        )

        f.write(
            f"non-tRF records: "
            f"{(df.label == 'non-tRF').sum():,}\n"
        )

        f.write(
            f"Global duplicate records: "
            f"{len(df) - df.sequence.nunique():,}\n"
        )

        f.write("\n")

        f.write(
            "tRF SUBCLASS COUNTS\n"
        )

        f.write("-" * 50 + "\n")

        counts = (
            df[df["label"] == "tRF"]
            ["subclass"]
            .value_counts()
        )

        for cls, count in counts.items():

            f.write(
                f"{cls}: {count:,}\n"
            )

        f.write("\n")

        f.write(
            "IMPORTANT OUTPUT FILES\n"
        )

        f.write("-" * 50 + "\n")

        for path in sorted(
            OUT_DIR.rglob("*")
        ):

            if path.is_file():

                f.write(
                    str(
                        path.relative_to(
                            OUT_DIR
                        )
                    ) + "\n"
                )

    log(
        f"Summary written to {summary_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    start = time.time()

    log("=" * 70)
    log("STARTING STAGE 1")
    log("=" * 70)

    # Check inputs
    required = [
        POSITIVE_ALL,
        NEGATIVE_ALL,
        *SUBTYPE_FILES.values()
    ]

    missing = [
        str(x)
        for x in required
        if not x.exists()
    ]

    if missing:

        log("ERROR: Missing input files:")

        for x in missing:
            log(x)

        sys.exit(1)

    # 1
    df, subtype_membership = dataset_audit()

    # 2
    duplicate_analysis(df)

    # 3
    feature_df = extract_features(df)

    # 4
    descriptive_statistics(
        feature_df
    )

    # 5
    binary_statistics(
        feature_df
    )

    # 6
    subclass_statistics(
        feature_df
    )

    # 7
    kmer_enrichment(
        feature_df
    )

    # 8
    feature_correlation(
        feature_df
    )

    # 9
    mutual_information_analysis(
        feature_df
    )

    # 10
    pca_analysis(
        feature_df
    )

    # 11
    umap_analysis(
        feature_df
    )

    # 12
    cd_hit_analysis()

    # 13
    generate_summary(
        feature_df
    )

    elapsed = time.time() - start

    log("=" * 70)
    log(
        f"STAGE 1 COMPLETE — "
        f"{elapsed / 3600:.2f} hours"
    )
    log("=" * 70)


if __name__ == "__main__":
    main()
