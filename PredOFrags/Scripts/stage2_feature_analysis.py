#!/usr/bin/env python3

"""
STAGE 2
============================================================
tRF / non-tRF FEATURE ENGINEERING AND STATISTICAL ANALYSIS

Input:
    cleaned_dataset/merged_pos_data/
    cleaned_dataset/negative_dataset/

No ML classifier training is performed here.

Outputs:
    stage2_features/
        01_dataset/
        02_features/
        03_statistics/
        04_feature_selection/
        05_PCA/
        06_UMAP/
        07_visualization_data/
        08_logs/
        stage2_summary.txt

Designed for large datasets on HPC.
============================================================
"""

from pathlib import Path
from collections import Counter
from itertools import product
from datetime import datetime
import math
import sys
import warnings

import numpy as np
import pandas as pd

from scipy.stats import (
    mannwhitneyu,
    kruskal,
    ks_2samp,
    chi2_contingency,
)

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path("/lustre/home/sklab202/sourav_ml/sheetal_ml")

INPUT_DIR = BASE / "cleaned_dataset"

POS_DIR = INPUT_DIR / "merged_pos_data"
NEG_DIR = INPUT_DIR / "negative_dataset"

OUT = BASE / "stage2_features"

DIRS = {
    "dataset": OUT / "01_dataset",
    "features": OUT / "02_features",
    "statistics": OUT / "03_statistics",
    "selection": OUT / "04_feature_selection",
    "pca": OUT / "05_PCA",
    "umap": OUT / "06_UMAP",
    "visual": OUT / "07_visualization_data",
    "logs": OUT / "08_logs",
}

for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)


RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

KMER_SIZES = [1, 2, 3, 4]

MAX_MI_SAMPLE = 100000
MAX_PCA_SAMPLE = 100000
MAX_UMAP_SAMPLE = 20000

CORRELATION_THRESHOLD = 0.90
VARIANCE_THRESHOLD = 1e-8

POSITION_N = 10


# ============================================================
# LOGGING
# ============================================================

LOG_FILE = DIRS["logs"] / "stage2.log"


def log(message):

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    line = f"[{timestamp}] {message}"

    print(line, flush=True)

    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ============================================================
# FASTA READER
# ============================================================

def read_fasta(path):

    records = []

    header = None
    seq_parts = []

    with open(path) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):

                if header is not None:
                    records.append(
                        (header, "".join(seq_parts))
                    )

                header = line
                seq_parts = []

            else:

                seq_parts.append(line)

        if header is not None:

            records.append(
                (header, "".join(seq_parts))
            )

    return records


# ============================================================
# SEQUENCE NORMALIZATION
# ============================================================

def normalize_sequence(seq):

    seq = seq.upper().replace("T", "U")

    # Remove whitespace
    seq = "".join(seq.split())

    return seq


# ============================================================
# FASTA FILE DEFINITIONS
# ============================================================

POSITIVE_SUBTYPES = {
    "3tRH": "3tRH",
    "5tRH": "5tRH",
    "leader_tRF": "leader",
    "other_tRF": "other",
    "tRF_1": "tRF-1",
    "tRF_3_all": "tRF-3",
    "tRF_5": "tRF-5",
}

POSITIVE_FILES = [
    "3tRH.fa",
    "5tRH.fa",
    "all_positive_merged_unique.fa",
    "leader_tRF.fa",
    "other_tRF.fa",
    "tRF_1.fa",
    "tRF_3_all.fa",
    "tRF_5.fa",
]

NEGATIVE_FILES = [
    "3tRH_Result.fa",
    "5tRH_Result.fa",
    "all_negative.fa",
    "databse_ncbi_negative_dataset_unique.fa",
    "leader_Result.fa",
    "other_Result.fa",
    "tRF_1_Result.fa",
    "tRF_3_Result.fa",
    "tRF_5_Result.fa",
]


# ============================================================
# STEP 1
# BUILD UNIQUE SUBTYPE MAP
# ============================================================

def build_subtype_map():

    log("Building positive tRF subtype map")

    subtype_membership = {}

    for filename, subtype in POSITIVE_SUBTYPES.items():

        fasta_file = POS_DIR / f"{filename}.fa"

        if not fasta_file.exists():

            log(f"WARNING: missing subtype file: {fasta_file}")

            continue

        records = read_fasta(fasta_file)

        count = 0

        for header, seq in records:

            seq = normalize_sequence(seq)

            if not seq:
                continue

            if seq not in subtype_membership:

                subtype_membership[seq] = subtype

            else:

                # This should be zero after cleaning
                if subtype_membership[seq] != subtype:

                    raise RuntimeError(
                        "Multiple subtype assignment detected after cleaning: "
                        f"{seq}"
                    )

            count += 1

        log(
            f"Subtype {subtype}: {count:,} records"
        )

    return subtype_membership


# ============================================================
# STEP 2
# BUILD MASTER DATASET
# ============================================================

def build_master_dataset():

    log("Reading cleaned positive and negative datasets")

    positive_file = POS_DIR / "all_positive_merged_unique.fa"
    negative_file = NEG_DIR / "all_negative.fa"

    if not positive_file.exists():

        raise FileNotFoundError(
            positive_file
        )

    if not negative_file.exists():

        raise FileNotFoundError(
            negative_file
        )

    subtype_map = build_subtype_map()

    records = []

    # --------------------------------------------------------
    # POSITIVE
    # --------------------------------------------------------

    positive_records = read_fasta(positive_file)

    log(
        f"Positive records read: "
        f"{len(positive_records):,}"
    )

    for i, (header, seq) in enumerate(positive_records):

        seq = normalize_sequence(seq)

        subtype = subtype_map.get(
            seq,
            "UNASSIGNED"
        )

        records.append({
            "sequence_id": f"POS_{i+1:08d}",
            "header": header,
            "sequence": seq,
            "label": "tRF",
            "label_numeric": 1,
            "subclass": subtype,
        })

    # --------------------------------------------------------
    # NEGATIVE
    # --------------------------------------------------------

    negative_records = read_fasta(negative_file)

    log(
        f"Negative records read: "
        f"{len(negative_records):,}"
    )

    for i, (header, seq) in enumerate(negative_records):

        seq = normalize_sequence(seq)

        records.append({
            "sequence_id": f"NEG_{i+1:08d}",
            "header": header,
            "sequence": seq,
            "label": "non-tRF",
            "label_numeric": 0,
            "subclass": "non-tRF",
        })

    df = pd.DataFrame(records)

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    log("Running dataset integrity checks")

    duplicates = df[
        df["sequence"].duplicated(
            keep=False
        )
    ]

    if len(duplicates) > 0:

        log(
            f"WARNING: {len(duplicates):,} "
            "duplicate records remain"
        )

    else:

        log("No duplicate sequences detected")

    cross_class = (
        df.groupby("sequence")["label"]
        .nunique()
    )

    cross_class_count = int(
        (cross_class > 1).sum()
    )

    log(
        "Cross-class identical sequences: "
        f"{cross_class_count}"
    )

    if cross_class_count > 0:

        raise RuntimeError(
            "Cross-class duplicate sequences remain."
        )

    unassigned = int(
        (df["subclass"] == "UNASSIGNED").sum()
    )

    log(
        f"Unassigned positive subtype sequences: "
        f"{unassigned:,}"
    )

    if unassigned > 0:

        log(
            "WARNING: Some positive sequences are "
            "not present in subtype FASTA files."
        )

    output = DIRS["dataset"] / "master_dataset.tsv"

    df.to_csv(
        output,
        sep="\t",
        index=False
    )

    log(
        f"Master dataset saved: {output}"
    )

    return df


# ============================================================
# BASIC SEQUENCE FEATURES
# ============================================================

def shannon_entropy(seq):

    if not seq:
        return 0.0

    counts = Counter(seq)

    n = len(seq)

    entropy = 0.0

    for count in counts.values():

        p = count / n

        entropy -= p * math.log2(p)

    return entropy


def sequence_complexity(seq):

    if not seq:
        return 0.0

    return len(set(seq)) / 4.0


def longest_homopolymer(seq):

    if not seq:
        return 0

    best = 1
    current = 1

    for i in range(1, len(seq)):

        if seq[i] == seq[i - 1]:

            current += 1

            best = max(best, current)

        else:

            current = 1

    return best


def count_homopolymers(seq, min_length=3):

    if not seq:
        return 0

    count = 0
    current = 1

    for i in range(1, len(seq)):

        if seq[i] == seq[i - 1]:

            current += 1

        else:

            if current >= min_length:
                count += 1

            current = 1

    if current >= min_length:
        count += 1

    return count


# ============================================================
# K-MER FEATURES
# ============================================================

def generate_kmers(k):

    return [
        "".join(x)
        for x in product("ACGU", repeat=k)
    ]


KMER_DICT = {
    k: generate_kmers(k)
    for k in KMER_SIZES
}


def kmer_frequency(seq, k):

    kmers = KMER_DICT[k]

    counts = dict.fromkeys(
        kmers,
        0
    )

    total = len(seq) - k + 1

    if total <= 0:

        return {
            f"k{k}_{mer}": 0.0
            for mer in kmers
        }

    for i in range(total):

        mer = seq[i:i+k]

        if mer in counts:

            counts[mer] += 1

    return {
        f"k{k}_{mer}": counts[mer] / total
        for mer in kmers
    }


# ============================================================
# TERMINAL FEATURES
# ============================================================

def terminal_features(seq):

    features = {}

    length = len(seq)

    # --------------------------------------------------------
    # Individual terminal nucleotides
    # --------------------------------------------------------

    for i in range(1, POSITION_N + 1):

        features[
            f"five_prime_pos_{i}"
        ] = (
            seq[i-1]
            if length >= i
            else "N"
        )

        features[
            f"three_prime_pos_{i}"
        ] = (
            seq[-i]
            if length >= i
            else "N"
        )

    # --------------------------------------------------------
    # Terminal composition
    # --------------------------------------------------------

    for region_name, region in [
        ("five_prime_5nt", seq[:5]),
        ("three_prime_5nt", seq[-5:]),
        ("five_prime_10nt", seq[:10]),
        ("three_prime_10nt", seq[-10:]),
    ]:

        region_length = len(region)

        for nt in "ACGU":

            features[
                f"{region_name}_{nt}_frac"
            ] = (
                region.count(nt) / region_length
                if region_length > 0
                else 0.0
            )

        features[
            f"{region_name}_GC"
        ] = (
            (region.count("G") + region.count("C"))
            / region_length
            if region_length > 0
            else 0.0
        )

        features[
            f"{region_name}_entropy"
        ] = shannon_entropy(region)

    # --------------------------------------------------------
    # First/last nucleotide indicators
    # --------------------------------------------------------

    for side, nucleotide in [
        ("five_prime", seq[0] if seq else "N"),
        ("three_prime", seq[-1] if seq else "N"),
    ]:

        for nt in "ACGU":

            features[
                f"{side}_{nt}"
            ] = int(
                nucleotide == nt
            )

    return features


# ============================================================
# POSITIONAL BINS
# ============================================================

def positional_bin_features(seq):

    features = {}

    n = len(seq)

    if n == 0:
        return features

    # Divide sequence into 5 normalized regions
    for bin_id in range(5):

        start = int(
            bin_id * n / 5
        )

        end = int(
            (bin_id + 1) * n / 5
        )

        region = seq[start:end]

        if not region:
            continue

        for nt in "ACGU":

            features[
                f"relative_bin_{bin_id+1}_{nt}"
            ] = (
                region.count(nt)
                / len(region)
            )

        features[
            f"relative_bin_{bin_id+1}_GC"
        ] = (
            region.count("G")
            + region.count("C")
        ) / len(region)

        features[
            f"relative_bin_{bin_id+1}_entropy"
        ] = shannon_entropy(region)

    return features


# ============================================================
# FULL FEATURE EXTRACTION
# ============================================================

def extract_features(df):

    log(
        "Extracting sequence features"
    )

    rows = []

    total = len(df)

    for idx, row in enumerate(
        df.itertuples(index=False),
        start=1
    ):

        seq = row.sequence

        length = len(seq)

        counts = {
            nt: seq.count(nt)
            for nt in "ACGU"
        }

        row_features = {
            "sequence_id": row.sequence_id,
            "label": row.label,
            "label_numeric": row.label_numeric,
            "subclass": row.subclass,
            "length": length,

            "A_count": counts["A"],
            "C_count": counts["C"],
            "G_count": counts["G"],
            "U_count": counts["U"],

            "A_fraction": counts["A"] / length
            if length else 0,

            "C_fraction": counts["C"] / length
            if length else 0,

            "G_fraction": counts["G"] / length
            if length else 0,

            "U_fraction": counts["U"] / length
            if length else 0,

            "GC_fraction": (
                counts["G"] + counts["C"]
            ) / length if length else 0,

            "AU_fraction": (
                counts["A"] + counts["U"]
            ) / length if length else 0,

            "GC_skew": (
                (counts["G"] - counts["C"])
                / (counts["G"] + counts["C"])
                if (counts["G"] + counts["C"]) > 0
                else 0
            ),

            "AU_skew": (
                (counts["A"] - counts["U"])
                / (counts["A"] + counts["U"])
                if (counts["A"] + counts["U"]) > 0
                else 0
            ),

            "purine_fraction": (
                counts["A"] + counts["G"]
            ) / length if length else 0,

            "pyrimidine_fraction": (
                counts["C"] + counts["U"]
            ) / length if length else 0,

            "shannon_entropy": shannon_entropy(seq),

            "sequence_complexity": sequence_complexity(seq),

            "unique_nucleotide_count":
                len(set(seq)),

            "longest_homopolymer":
                longest_homopolymer(seq),

            "homopolymer_runs_ge3":
                count_homopolymers(seq, 3),

            "homopolymer_runs_ge4":
                count_homopolymers(seq, 4),

            "homopolymer_runs_ge5":
                count_homopolymers(seq, 5),
        }

        # Terminal features
        row_features.update(
            terminal_features(seq)
        )

        # Relative positional features
        row_features.update(
            positional_bin_features(seq)
        )

        # k-mers
        for k in KMER_SIZES:

            row_features.update(
                kmer_frequency(seq, k)
            )

        rows.append(row_features)

        if idx % 25000 == 0:

            log(
                f"Feature extraction: "
                f"{idx:,}/{total:,}"
            )

    features = pd.DataFrame(rows)

    output = (
        DIRS["features"]
        / "complete_feature_matrix.tsv"
    )

    features.to_csv(
        output,
        sep="\t",
        index=False
    )

    log(
        f"Feature matrix saved: "
        f"{features.shape[0]:,} rows × "
        f"{features.shape[1]:,} columns"
    )

    return features


# ============================================================
# NUMERIC FEATURE MATRIX
# ============================================================

def get_numeric_features(features):

    excluded = {
        "sequence_id",
        "label",
        "subclass",
    }

    numeric_cols = [
        c for c in features.columns
        if c not in excluded
        and pd.api.types.is_numeric_dtype(
            features[c]
        )
    ]

    numeric = features[
        ["sequence_id", "label", "subclass"]
        + numeric_cols
    ].copy()

    output = (
        DIRS["features"]
        / "numeric_feature_matrix.tsv"
    )

    numeric.to_csv(
        output,
        sep="\t",
        index=False
    )

    return numeric, numeric_cols


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

def descriptive_statistics(
    features,
    numeric_cols
):

    log(
        "Calculating descriptive statistics"
    )

    groups = [
        ("label", "tRF_vs_non_tRF"),
        ("subclass", "subclass"),
    ]

    all_results = []

    for group_col, analysis_name in groups:

        for group, group_df in features.groupby(
            group_col
        ):

            for feature in numeric_cols:

                values = pd.to_numeric(
                    group_df[feature],
                    errors="coerce"
                ).dropna()

                if len(values) == 0:
                    continue

                all_results.append({
                    "analysis": analysis_name,
                    "group": group,
                    "feature": feature,
                    "n": len(values),
                    "mean": values.mean(),
                    "std": values.std(),
                    "median": values.median(),
                    "q25": values.quantile(0.25),
                    "q75": values.quantile(0.75),
                    "min": values.min(),
                    "max": values.max(),
                })

    result = pd.DataFrame(all_results)

    output = (
        DIRS["statistics"]
        / "descriptive_statistics.tsv"
    )

    result.to_csv(
        output,
        sep="\t",
        index=False
    )


# ============================================================
# CLAFF'S DELTA
# ============================================================

def cliffs_delta(x, y):

    x = np.asarray(x)
    y = np.asarray(y)

    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    # Efficient rank-based calculation
    combined = np.concatenate([x, y])

    ranks = pd.Series(
        combined
    ).rank(
        method="average"
    ).values

    nx = len(x)
    ny = len(y)

    rank_x = ranks[:nx].sum()

    U = (
        rank_x
        - nx * (nx + 1) / 2
    )

    delta = (
        2 * U / (nx * ny)
        - 1
    )

    return delta


# ============================================================
# BENJAMINI-HOCHBERG FDR
# ============================================================

def bh_fdr(pvalues):

    pvalues = np.asarray(
        pvalues,
        dtype=float
    )

    n = len(pvalues)

    order = np.argsort(
        pvalues
    )

    ranked = pvalues[order]

    adjusted = np.empty(n)

    cumulative = 1.0

    for i in range(n - 1, -1, -1):

        rank = i + 1

        value = (
            ranked[i] * n / rank
        )

        cumulative = min(
            cumulative,
            value
        )

        adjusted[i] = cumulative

    result = np.empty(n)

    result[order] = adjusted

    return np.clip(
        result,
        0,
        1
    )


# ============================================================
# BINARY STATISTICS
# ============================================================

def binary_statistics(
    features,
    numeric_cols
):

    log(
        "Running binary tRF vs non-tRF statistics"
    )

    tRF = features[
        features["label"] == "tRF"
    ]

    non = features[
        features["label"] == "non-tRF"
    ]

    results = []

    for feature in numeric_cols:

        x = pd.to_numeric(
            tRF[feature],
            errors="coerce"
        ).dropna().values

        y = pd.to_numeric(
            non[feature],
            errors="coerce"
        ).dropna().values

        if len(x) == 0 or len(y) == 0:
            continue

        try:

            mw = mannwhitneyu(
                x,
                y,
                alternative="two-sided"
            )

            ks = ks_2samp(
                x,
                y
            )

            delta = cliffs_delta(
                x,
                y
            )

            results.append({
                "feature": feature,
                "n_tRF": len(x),
                "n_non_tRF": len(y),
                "median_tRF": np.median(x),
                "median_non_tRF": np.median(y),
                "mean_tRF": np.mean(x),
                "mean_non_tRF": np.mean(y),
                "mannwhitney_U": mw.statistic,
                "mannwhitney_p": mw.pvalue,
                "ks_statistic": ks.statistic,
                "ks_p": ks.pvalue,
                "cliffs_delta": delta,
            })

        except Exception:

            continue

    result = pd.DataFrame(results)

    result[
        "mannwhitney_fdr"
    ] = bh_fdr(
        result["mannwhitney_p"]
    )

    result[
        "ks_fdr"
    ] = bh_fdr(
        result["ks_p"]
    )

    result = result.sort_values(
        "mannwhitney_fdr"
    )

    output = (
        DIRS["statistics"]
        / "binary_feature_statistics.tsv"
    )

    result.to_csv(
        output,
        sep="\t",
        index=False
    )

    return result


# ============================================================
# MULTICLASS KRUSKAL-WALLIS
# ============================================================

def subclass_kruskal(
    features,
    numeric_cols
):

    log(
        "Running 7-class Kruskal-Wallis tests"
    )

    positive = features[
        features["label"] == "tRF"
    ].copy()

    positive = positive[
        positive["subclass"] != "UNASSIGNED"
    ]

    groups = {
        name: group
        for name, group
        in positive.groupby("subclass")
    }

    results = []

    for feature in numeric_cols:

        arrays = []

        for group in groups.values():

            values = pd.to_numeric(
                group[feature],
                errors="coerce"
            ).dropna().values

            if len(values) > 0:

                arrays.append(values)

        if len(arrays) < 2:
            continue

        try:

            stat, p = kruskal(
                *arrays
            )

            results.append({
                "feature": feature,
                "H_statistic": stat,
                "p_value": p,
            })

        except Exception:

            continue

    result = pd.DataFrame(results)

    result["FDR"] = bh_fdr(
        result["p_value"]
    )

    result = result.sort_values(
        "FDR"
    )

    output = (
        DIRS["statistics"]
        / "subclass_kruskal_wallis.tsv"
    )

    result.to_csv(
        output,
        sep="\t",
        index=False
    )

    return result


# ============================================================
# PAIRWISE SUBTYPE TESTS
# ============================================================

def pairwise_subclass_statistics(
    features,
    numeric_cols
):

    log(
        "Running pairwise subtype tests"
    )

    positive = features[
        features["label"] == "tRF"
    ]

    positive = positive[
        positive["subclass"] != "UNASSIGNED"
    ]

    subtypes = sorted(
        positive["subclass"].unique()
    )

    results = []

    for i in range(len(subtypes)):

        for j in range(i + 1, len(subtypes)):

            s1 = subtypes[i]
            s2 = subtypes[j]

            d1 = positive[
                positive["subclass"] == s1
            ]

            d2 = positive[
                positive["subclass"] == s2
            ]

            for feature in numeric_cols:

                x = pd.to_numeric(
                    d1[feature],
                    errors="coerce"
                ).dropna().values

                y = pd.to_numeric(
                    d2[feature],
                    errors="coerce"
                ).dropna().values

                if len(x) == 0 or len(y) == 0:
                    continue

                try:

                    mw = mannwhitneyu(
                        x,
                        y,
                        alternative="two-sided"
                    )

                    delta = cliffs_delta(
                        x,
                        y
                    )

                    results.append({
                        "subclass_1": s1,
                        "subclass_2": s2,
                        "feature": feature,
                        "mannwhitney_U": mw.statistic,
                        "p_value": mw.pvalue,
                        "cliffs_delta": delta,
                    })

                except Exception:

                    continue

    result = pd.DataFrame(results)

    if len(result) > 0:

        result["FDR"] = bh_fdr(
            result["p_value"]
        )

        result = result.sort_values(
            "FDR"
        )

    output = (
        DIRS["statistics"]
        / "subclass_pairwise_statistics.tsv"
    )

    result.to_csv(
        output,
        sep="\t",
        index=False
    )


# ============================================================
# K-MER ENRICHMENT
# ============================================================

def kmer_enrichment(
    features
):

    log(
        "Running k-mer enrichment analysis"
    )

    results = []

    binary = features["label_numeric"].values

    for k in KMER_SIZES:

        prefix = f"k{k}_"

        kmer_cols = [
            c for c in features.columns
            if c.startswith(prefix)
        ]

        for feature in kmer_cols:

            # Presence / absence
            presence = (
                features[feature] > 0
            ).astype(int).values

            table = pd.crosstab(
                binary,
                presence
            )

            # Ensure 2 x 2
            table = table.reindex(
                index=[0, 1],
                columns=[0, 1],
                fill_value=0
            )

            try:

                chi2, p, _, _ = (
                    chi2_contingency(
                        table
                    )
                )

                a = table.loc[1, 1]
                b = table.loc[1, 0]
                c = table.loc[0, 1]
                d = table.loc[0, 0]

                # Haldane correction
                OR = (
                    (a + 0.5) * (d + 0.5)
                    /
                    ((b + 0.5) * (c + 0.5))
                )

                log2OR = np.log2(OR)

                results.append({
                    "k": k,
                    "feature": feature,
                    "tRF_present": int(a),
                    "tRF_absent": int(b),
                    "non_tRF_present": int(c),
                    "non_tRF_absent": int(d),
                    "odds_ratio": OR,
                    "log2_odds_ratio": log2OR,
                    "chi2": chi2,
                    "p_value": p,
                })

            except Exception:

                continue

    result = pd.DataFrame(results)

    result["FDR"] = bh_fdr(
        result["p_value"]
    )

    result = result.sort_values(
        "FDR"
    )

    output = (
        DIRS["statistics"]
        / "kmer_enrichment.tsv"
    )

    result.to_csv(
        output,
        sep="\t",
        index=False
    )

    # Separate files for plotting
    for k in KMER_SIZES:

        sub = result[
            result["k"] == k
        ]

        sub.to_csv(
            DIRS["visual"]
            / f"kmer_enrichment_k{k}.tsv",
            sep="\t",
            index=False
        )


# ============================================================
# MUTUAL INFORMATION
# ============================================================

def mutual_information(
    features,
    numeric_cols
):

    log(
        "Calculating mutual information"
    )

    n = len(features)

    if n > MAX_MI_SAMPLE:

        sample_idx = np.random.RandomState(
            RANDOM_SEED
        ).choice(
            n,
            size=MAX_MI_SAMPLE,
            replace=False
        )

        sample = features.iloc[
            sample_idx
        ].copy()

    else:

        sample = features.copy()

    X = sample[
        numeric_cols
    ].astype(float)

    y_binary = sample[
        "label_numeric"
    ].values

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    # Remove constant columns
    valid = [
        c for c in numeric_cols
        if X[c].nunique() > 1
    ]

    X = X[valid]

    mi_binary = mutual_info_classif(
        X.values,
        y_binary,
        random_state=RANDOM_SEED
    )

    binary_result = pd.DataFrame({
        "feature": valid,
        "mutual_information": mi_binary
    }).sort_values(
        "mutual_information",
        ascending=False
    )

    binary_result.to_csv(
        DIRS["statistics"]
        / "mutual_information_binary.tsv",
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # Multiclass MI
    # --------------------------------------------------------

    positive = sample[
        sample["label"] == "tRF"
    ].copy()

    positive = positive[
        positive["subclass"] != "UNASSIGNED"
    ]

    if len(positive) > 0:

        Xp = positive[
            valid
        ].astype(float).fillna(0)

        y_sub = (
            positive["subclass"]
            .astype("category")
            .cat.codes
            .values
        )

        valid2 = [
            c for c in valid
            if Xp[c].nunique() > 1
        ]

        mi_sub = mutual_info_classif(
            Xp[valid2].values,
            y_sub,
            random_state=RANDOM_SEED
        )

        subclass_result = pd.DataFrame({
            "feature": valid2,
            "mutual_information":
                mi_sub
        }).sort_values(
            "mutual_information",
            ascending=False
        )

        subclass_result.to_csv(
            DIRS["statistics"]
            / "mutual_information_subclass.tsv",
            sep="\t",
            index=False
        )


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def correlation_analysis(
    features,
    numeric_cols
):

    log(
        "Calculating Pearson and Spearman correlations"
    )

    X = features[
        numeric_cols
    ].astype(float)

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    # --------------------------------------------------------
    # Pearson
    # --------------------------------------------------------

    pearson = X.corr(
        method="pearson"
    )

    pearson.to_csv(
        DIRS["statistics"]
        / "feature_correlation_pearson.tsv",
        sep="\t"
    )

    # --------------------------------------------------------
    # Spearman
    # --------------------------------------------------------

    spearman = X.corr(
        method="spearman"
    )

    spearman.to_csv(
        DIRS["statistics"]
        / "feature_correlation_spearman.tsv",
        sep="\t"
    )

    # --------------------------------------------------------
    # Long format
    # --------------------------------------------------------

    pairs = []

    cols = pearson.columns.tolist()

    for i in range(len(cols)):

        for j in range(i + 1, len(cols)):

            p = pearson.iloc[i, j]
            s = spearman.iloc[i, j]

            pairs.append({
                "feature_1": cols[i],
                "feature_2": cols[j],
                "pearson": p,
                "spearman": s,
                "abs_pearson": abs(p),
                "abs_spearman": abs(s),
            })

    long_df = pd.DataFrame(pairs)

    long_df = long_df.sort_values(
        "abs_pearson",
        ascending=False
    )

    long_df.to_csv(
        DIRS["statistics"]
        / "feature_correlation_long.tsv",
        sep="\t",
        index=False
    )

    return pearson


# ============================================================
# VARIANCE FILTER
# ============================================================

def variance_filter(
    features,
    numeric_cols
):

    log(
        "Running variance filtering"
    )

    X = features[
        numeric_cols
    ].astype(float)

    variances = X.var()

    result = pd.DataFrame({
        "feature": variances.index,
        "variance": variances.values,
        "retain_variance":
            variances.values
            > VARIANCE_THRESHOLD
    })

    result = result.sort_values(
        "variance"
    )

    result.to_csv(
        DIRS["selection"]
        / "variance_filter.tsv",
        sep="\t",
        index=False
    )

    retained = result.loc[
        result["retain_variance"],
        "feature"
    ].tolist()

    return retained


# ============================================================
# CORRELATION-BASED FEATURE PRUNING
# ============================================================

def correlation_pruning(
    features,
    numeric_cols,
    mi_binary=None
):

    log(
        "Running correlation-based feature pruning"
    )

    X = features[
        numeric_cols
    ].astype(float)

    X = X.fillna(0)

    corr = X.corr(
        method="spearman"
    ).abs()

    # Use MI ranking if available.
    if mi_binary is not None:

        ranking = (
            mi_binary
            .set_index("feature")
            ["mutual_information"]
            .to_dict()
        )

    else:

        ranking = {
            c: 0
            for c in numeric_cols
        }

    ordered = sorted(
        numeric_cols,
        key=lambda x: ranking.get(x, 0),
        reverse=True
    )

    selected = []

    removed = []

    for feature in ordered:

        keep = True

        for chosen in selected:

            if corr.loc[
                feature,
                chosen
            ] >= CORRELATION_THRESHOLD:

                keep = False

                removed.append({
                    "feature_removed": feature,
                    "feature_retained": chosen,
                    "absolute_spearman":
                        corr.loc[
                            feature,
                            chosen
                        ],
                    "reason":
                        "high_correlation"
                })

                break

        if keep:

            selected.append(feature)

    removed_df = pd.DataFrame(
        removed
    )

    removed_df.to_csv(
        DIRS["selection"]
        / "correlation_removed_features.tsv",
        sep="\t",
        index=False
    )

    selected_df = pd.DataFrame({
        "feature": selected,
        "selection_order":
            range(1, len(selected) + 1)
    })

    selected_df.to_csv(
        DIRS["selection"]
        / "correlation_pruned_features.tsv",
        sep="\t",
        index=False
    )

    log(
        f"Correlation pruning: "
        f"{len(numeric_cols):,} -> "
        f"{len(selected):,} features"
    )

    return selected


# ============================================================
# COMBINED FEATURE RANKING
# ============================================================

def combined_feature_ranking(
    binary_stats,
    kruskal_file,
    mi_file,
    variance_file,
    selected_features
):

    log(
        "Creating combined feature ranking"
    )

    ranking = pd.DataFrame({
        "feature": selected_features
    })

    # Binary significance
    binary = binary_stats[
        [
            "feature",
            "mannwhitney_fdr",
            "cliffs_delta",
            "ks_fdr"
        ]
    ].copy()

    ranking = ranking.merge(
        binary,
        on="feature",
        how="left"
    )

    # Kruskal
    kruskal = pd.read_csv(
        kruskal_file,
        sep="\t"
    )

    ranking = ranking.merge(
        kruskal[
            [
                "feature",
                "FDR"
            ]
        ].rename(
            columns={
                "FDR":
                    "subclass_kruskal_fdr"
            }
        ),
        on="feature",
        how="left"
    )

    # MI
    mi = pd.read_csv(
        mi_file,
        sep="\t"
    )

    ranking = ranking.merge(
        mi,
        on="feature",
        how="left"
    )

    # Variance
    variance = pd.read_csv(
        variance_file,
        sep="\t"
    )

    ranking = ranking.merge(
        variance[
            [
                "feature",
                "variance"
            ]
        ],
        on="feature",
        how="left"
    )

    # Composite exploratory rank
    ranking["binary_significant"] = (
        ranking["mannwhitney_fdr"] < 0.05
    ).astype(int)

    ranking["subclass_significant"] = (
        ranking["subclass_kruskal_fdr"] < 0.05
    ).astype(int)

    ranking["mi_rank"] = (
        ranking[
            "mutual_information"
        ]
        .rank(
            ascending=False,
            method="average"
        )
    )

    ranking["binary_effect_abs"] = (
        ranking["cliffs_delta"]
        .abs()
    )

    ranking = ranking.sort_values(
        [
            "binary_significant",
            "subclass_significant",
            "mutual_information",
            "binary_effect_abs"
        ],
        ascending=[
            False,
            False,
            False,
            False
        ]
    )

    ranking[
        "exploratory_rank"
    ] = range(
        1,
        len(ranking) + 1
    )

    ranking.to_csv(
        DIRS["selection"]
        / "combined_feature_ranking.tsv",
        sep="\t",
        index=False
    )

    return ranking


# ============================================================
# PCA
# ============================================================

def run_pca(
    features,
    numeric_cols
):

    log("Running PCA")

    n = len(features)

    if n > MAX_PCA_SAMPLE:

        rng = np.random.RandomState(
            RANDOM_SEED
        )

        idx = rng.choice(
            n,
            MAX_PCA_SAMPLE,
            replace=False
        )

        sample = features.iloc[
            idx
        ].copy()

    else:

        sample = features.copy()

    X = sample[
        numeric_cols
    ].astype(float)

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    # Remove zero variance
    valid = [
        c for c in X.columns
        if X[c].std() > 0
    ]

    X = X[valid]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    n_components = min(
        20,
        X_scaled.shape[0],
        X_scaled.shape[1]
    )

    pca = PCA(
        n_components=n_components,
        random_state=RANDOM_SEED
    )

    coords = pca.fit_transform(
        X_scaled
    )

    variance = pd.DataFrame({
        "PC": [
            f"PC{i+1}"
            for i in range(
                n_components
            )
        ],
        "explained_variance_ratio":
            pca.explained_variance_ratio_,
    })

    variance[
        "cumulative_variance"
    ] = variance[
        "explained_variance_ratio"
    ].cumsum()

    variance.to_csv(
        DIRS["pca"]
        / "PCA_explained_variance.tsv",
        sep="\t",
        index=False
    )

    loading = pd.DataFrame(
        pca.components_.T,
        index=valid,
        columns=[
            f"PC{i+1}"
            for i in range(
                n_components
            )
        ]
    )

    loading.index.name = "feature"

    loading.to_csv(
        DIRS["pca"]
        / "PCA_feature_loadings.tsv",
        sep="\t"
    )

    coordinate_df = pd.DataFrame(
        coords,
        columns=[
            f"PC{i+1}"
            for i in range(
                n_components
            )
        ]
    )

    coordinate_df.insert(
        0,
        "sequence_id",
        sample["sequence_id"].values
    )

    coordinate_df.insert(
        1,
        "label",
        sample["label"].values
    )

    coordinate_df.insert(
        2,
        "subclass",
        sample["subclass"].values
    )

    coordinate_df.to_csv(
        DIRS["visual"]
        / "PCA_coordinates.tsv",
        sep="\t",
        index=False
    )

    log(
        "PCA completed"
    )


# ============================================================
# UMAP
# ============================================================

def run_umap(
    features,
    numeric_cols
):

    try:

        import umap

    except ImportError:

        log(
            "umap-learn not installed. "
            "Skipping UMAP."
        )

        with open(
            DIRS["umap"]
            / "UMAP_status.txt",
            "w"
        ) as f:

            f.write(
                "SKIPPED: umap-learn not installed\n"
            )

        return

    log("Running UMAP")

    n = len(features)

    sample_size = min(
        MAX_UMAP_SAMPLE,
        n
    )

    rng = np.random.RandomState(
        RANDOM_SEED
    )

    idx = rng.choice(
        n,
        sample_size,
        replace=False
    )

    sample = features.iloc[
        idx
    ].copy()

    X = sample[
        numeric_cols
    ].astype(float)

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    valid = [
        c for c in X.columns
        if X[c].std() > 0
    ]

    X = X[valid]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.2,
        metric="euclidean",
        random_state=RANDOM_SEED,
        n_components=2,
    )

    embedding = reducer.fit_transform(
        X_scaled
    )

    result = pd.DataFrame({
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

    result.to_csv(
        DIRS["visual"]
        / "UMAP_coordinates.tsv",
        sep="\t",
        index=False
    )

    result.to_csv(
        DIRS["umap"]
        / "UMAP_coordinates.tsv",
        sep="\t",
        index=False
    )

    with open(
        DIRS["umap"]
        / "UMAP_status.txt",
        "w"
    ) as f:

        f.write(
            f"Completed\n"
            f"n={sample_size}\n"
            f"n_neighbors=30\n"
            f"min_dist=0.2\n"
            f"metric=euclidean\n"
            f"random_seed={RANDOM_SEED}\n"
        )

    log(
        "UMAP completed"
    )


# ============================================================
# VISUALIZATION DATA
# ============================================================

def create_visualization_tables(
    features
):

    log(
        "Creating visualization-ready tables"
    )

    # --------------------------------------------------------
    # Length
    # --------------------------------------------------------

    features[
        [
            "sequence_id",
            "label",
            "subclass",
            "length"
        ]
    ].to_csv(
        DIRS["visual"]
        / "length_distribution_data.tsv",
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # Composition
    # --------------------------------------------------------

    composition_cols = [
        "sequence_id",
        "label",
        "subclass",
        "A_fraction",
        "C_fraction",
        "G_fraction",
        "U_fraction",
        "GC_fraction",
        "AU_fraction",
        "GC_skew",
        "AU_skew",
    ]

    features[
        composition_cols
    ].to_csv(
        DIRS["visual"]
        / "composition_data.tsv",
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # Terminal features
    # --------------------------------------------------------

    terminal_cols = [
        c for c in features.columns
        if (
            c.startswith("five_prime_")
            or c.startswith("three_prime_")
        )
    ]

    features[
        [
            "sequence_id",
            "label",
            "subclass"
        ] + terminal_cols
    ].to_csv(
        DIRS["visual"]
        / "terminal_features.tsv",
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # Basic complexity
    # --------------------------------------------------------

    complexity_cols = [
        "sequence_id",
        "label",
        "subclass",
        "length",
        "shannon_entropy",
        "sequence_complexity",
        "unique_nucleotide_count",
        "longest_homopolymer",
        "homopolymer_runs_ge3",
        "homopolymer_runs_ge4",
        "homopolymer_runs_ge5",
    ]

    features[
        complexity_cols
    ].to_csv(
        DIRS["visual"]
        / "complexity_data.tsv",
        sep="\t",
        index=False
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

def write_summary(
    features,
    numeric_cols,
    binary_stats,
    kruskal_stats,
    selected_features,
):

    summary_file = (
        OUT / "stage2_summary.txt"
    )

    with open(
        summary_file,
        "w"
    ) as f:

        f.write(
            "STAGE 2 — tRF / non-tRF "
            "FEATURE ENGINEERING AND STATISTICAL ANALYSIS\n"
        )

        f.write(
            "=" * 75 + "\n\n"
        )

        f.write(
            f"Analysis date: "
            f"{datetime.now()}\n"
        )

        f.write(
            f"Base directory: "
            f"{BASE}\n"
        )

        f.write(
            f"Input directory: "
            f"{INPUT_DIR}\n"
        )

        f.write(
            f"Output directory: "
            f"{OUT}\n\n"
        )

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        f.write(
            "DATASET\n"
        )

        f.write(
            "-" * 50 + "\n"
        )

        f.write(
            f"Total sequences: "
            f"{len(features):,}\n"
        )

        f.write(
            f"tRF sequences: "
            f"{(features.label == 'tRF').sum():,}\n"
        )

        f.write(
            f"non-tRF sequences: "
            f"{(features.label == 'non-tRF').sum():,}\n"
        )

        f.write(
            f"Numeric features: "
            f"{len(numeric_cols):,}\n"
        )

        f.write(
            f"Correlation-pruned features: "
            f"{len(selected_features):,}\n\n"
        )

        # ----------------------------------------------------
        # Subclasses
        # ----------------------------------------------------

        f.write(
            "tRF SUBCLASSES\n"
        )

        f.write(
            "-" * 50 + "\n"
        )

        subclass_counts = (
            features[
                features.label == "tRF"
            ]
            ["subclass"]
            .value_counts()
        )

        for subtype, count in (
            subclass_counts.items()
        ):

            f.write(
                f"{subtype}: {count:,}\n"
            )

        f.write("\n")

        # ----------------------------------------------------
        # Binary significant features
        # ----------------------------------------------------

        f.write(
            "BINARY SIGNIFICANT FEATURES\n"
        )

        f.write(
            "-" * 50 + "\n"
        )

        binary_sig = binary_stats[
            binary_stats[
                "mannwhitney_fdr"
            ] < 0.05
        ]

        f.write(
            f"FDR < 0.05: "
            f"{len(binary_sig):,}\n"
        )

        f.write(
            f"FDR < 0.01: {(binary_stats['mannwhitney_fdr'] < 0.01).sum():,}\n"
        )

        f.write("\n")

        # ----------------------------------------------------
        # Multiclass significant
        # ----------------------------------------------------

        f.write(
            "MULTICLASS SIGNIFICANT FEATURES\n"
        )

        f.write(
            "-" * 50 + "\n"
        )

        kw_sig = kruskal_stats[
            kruskal_stats["FDR"] < 0.05
        ]

        f.write(
            f"Kruskal-Wallis FDR < 0.05: "
            f"{len(kw_sig):,}\n"
        )

        f.write("\n")

        # ----------------------------------------------------
        # Top features
        # ----------------------------------------------------

        f.write(
            "TOP BINARY FEATURES\n"
        )

        f.write(
            "-" * 50 + "\n"
        )

        for feature in (
            binary_stats
            .head(20)["feature"]
            .tolist()
        ):

            f.write(
                f"{feature}\n"
            )

        f.write("\n")

        f.write(
            "TOP CORRELATION-PRUNED FEATURES\n"
        )

        f.write(
            "-" * 50 + "\n"
        )

        for feature in selected_features[
            :50
        ]:

            f.write(
                f"{feature}\n"
            )

        f.write("\n")

        f.write(
            "IMPORTANT NOTE\n"
        )

        f.write(
            "-" * 50 + "\n"
        )

        f.write(
            "Feature significance and ranking here are "
            "exploratory dataset-level analyses. "
            "For final predictive modeling, feature "
            "selection thresholds must be recomputed "
            "using training data only to prevent "
            "information leakage.\n"
        )

    log(
        f"Summary saved: {summary_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    start = datetime.now()

    log("=" * 75)
    log("STARTING STAGE 2")
    log("=" * 75)

    log(
        f"Python version: {sys.version}"
    )

    log(
        f"Base directory: {BASE}"
    )

    log(
        f"Input directory: {INPUT_DIR}"
    )

    # --------------------------------------------------------
    # 1. Dataset
    # --------------------------------------------------------

    dataset = build_master_dataset()

    # --------------------------------------------------------
    # 2. Features
    # --------------------------------------------------------

    features = extract_features(
        dataset
    )

    numeric, numeric_cols = (
        get_numeric_features(
            features
        )
    )

    log(
        f"Numeric feature count: "
        f"{len(numeric_cols):,}"
    )

    # --------------------------------------------------------
    # 3. Descriptive statistics
    # --------------------------------------------------------

    descriptive_statistics(
        features,
        numeric_cols
    )

    # --------------------------------------------------------
    # 4. Binary statistics
    # --------------------------------------------------------

    binary_stats = binary_statistics(
        features,
        numeric_cols
    )

    # --------------------------------------------------------
    # 5. Multiclass statistics
    # --------------------------------------------------------

    kruskal_stats = subclass_kruskal(
        features,
        numeric_cols
    )

    pairwise_subclass_statistics(
        features,
        numeric_cols
    )

    # --------------------------------------------------------
    # 6. k-mer enrichment
    # --------------------------------------------------------

    kmer_enrichment(
        features
    )

    # --------------------------------------------------------
    # 7. Mutual information
    # --------------------------------------------------------

    mutual_information(
        features,
        numeric_cols
    )

    # --------------------------------------------------------
    # 8. Correlation
    # --------------------------------------------------------

    correlation_analysis(
        features,
        numeric_cols
    )

    # --------------------------------------------------------
    # 9. Variance filtering
    # --------------------------------------------------------

    variance_features = variance_filter(
        features,
        numeric_cols
    )

    # --------------------------------------------------------
    # 10. MI table
    # --------------------------------------------------------

    mi_file = (
        DIRS["statistics"]
        / "mutual_information_binary.tsv"
    )

    if mi_file.exists():

        mi_binary = pd.read_csv(
            mi_file,
            sep="\t"
        )

    else:

        mi_binary = None

    # --------------------------------------------------------
    # 11. Correlation pruning
    # --------------------------------------------------------

    selected_features = (
        correlation_pruning(
            features,
            variance_features,
            mi_binary
        )
    )

    # --------------------------------------------------------
    # 12. Combined ranking
    # --------------------------------------------------------

    combined_feature_ranking(
        binary_stats,
        DIRS["statistics"]
        / "subclass_kruskal_wallis.tsv",
        mi_file,
        DIRS["selection"]
        / "variance_filter.tsv",
        selected_features
    )

    # --------------------------------------------------------
    # 13. PCA
    # --------------------------------------------------------

    run_pca(
        features,
        numeric_cols
    )

    # --------------------------------------------------------
    # 14. UMAP
    # --------------------------------------------------------

    run_umap(
        features,
        numeric_cols
    )

    # --------------------------------------------------------
    # 15. Visualization tables
    # --------------------------------------------------------

    create_visualization_tables(
        features
    )

    # --------------------------------------------------------
    # 16. Summary
    # --------------------------------------------------------

    write_summary(
        features,
        numeric_cols,
        binary_stats,
        kruskal_stats,
        selected_features,
    )

    elapsed = (
        datetime.now() - start
    )

    log("=" * 75)
    log("STAGE 2 COMPLETED")
    log(f"Runtime: {elapsed}")
    log("=" * 75)


if __name__ == "__main__":

    main()
