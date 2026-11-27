#!/usr/bin/env python3

from pathlib import Path
from collections import OrderedDict

# ============================================================
# INPUT FILES
# ============================================================

EXTERNAL_FILE = Path("base_external.fa")
POSITIVE_FILE = Path("positive_trf_merged_unique.fa")

# ============================================================
# OUTPUT FILES
# ============================================================

OUT_DIR = Path("positive_external_comparison")
OUT_DIR.mkdir(exist_ok=True)

COMMON_EXTERNAL = OUT_DIR / "common_external_sequences.fa"
COMMON_POSITIVE = OUT_DIR / "common_positive_sequences.fa"
EXTERNAL_ONLY = OUT_DIR / "external_only_sequences.fa"
POSITIVE_ONLY = OUT_DIR / "positive_only_sequences.fa"
SUMMARY_FILE = OUT_DIR / "comparison_summary.txt"
COMMON_TABLE = OUT_DIR / "common_sequence_mapping.tsv"


def read_fasta(filename):
    """Read FASTA and retain headers and sequences."""
    records = OrderedDict()
    header = None
    sequence_parts = []

    with open(filename) as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    records[header] = "".join(sequence_parts).upper()

                header = line[1:]
                sequence_parts = []
            else:
                sequence_parts.append(line)

        if header is not None:
            records[header] = "".join(sequence_parts).upper()

    return records


def normalize_sequence(sequence):
    """Normalize DNA/RNA alphabet for sequence comparison."""
    return sequence.upper().replace("U", "T")


# ============================================================
# LOAD FILES
# ============================================================

if not EXTERNAL_FILE.exists():
    raise FileNotFoundError(f"Missing file: {EXTERNAL_FILE.resolve()}")

if not POSITIVE_FILE.exists():
    raise FileNotFoundError(f"Missing file: {POSITIVE_FILE.resolve()}")

external = read_fasta(EXTERNAL_FILE)
positive = read_fasta(POSITIVE_FILE)

# Map normalized sequence -> original FASTA records.
external_map = {}
positive_map = {}

for header, seq in external.items():
    external_map.setdefault(normalize_sequence(seq), []).append(
        (header, seq)
    )

for header, seq in positive.items():
    positive_map.setdefault(normalize_sequence(seq), []).append(
        (header, seq)
    )

external_sequences = set(external_map)
positive_sequences = set(positive_map)

# ============================================================
# COMPARE UNIQUE SEQUENCES
# ============================================================

common = external_sequences & positive_sequences
external_only = external_sequences - positive_sequences
positive_only = positive_sequences - external_sequences


def write_fasta(filename, sequence_set, sequence_map):
    """Write matching sequences with original headers."""
    with open(filename, "w") as out:
        for normalized_seq in sorted(sequence_set):
            for header, original_seq in sequence_map[normalized_seq]:
                out.write(f">{header}\n{original_seq}\n")


# Common sequences, retaining headers from each respective file.
write_fasta(COMMON_EXTERNAL, common, external_map)
write_fasta(COMMON_POSITIVE, common, positive_map)

# Sequences exclusive to either file.
write_fasta(EXTERNAL_ONLY, external_only, external_map)
write_fasta(POSITIVE_ONLY, positive_only, positive_map)

# ============================================================
# COMMON SEQUENCE MAPPING TABLE
# ============================================================

with open(COMMON_TABLE, "w") as out:
    out.write(
        "Normalized_Sequence\tExternal_Headers\tPositive_Headers\n"
    )

    for seq in sorted(common):
        external_headers = ";".join(
            header for header, _ in external_map[seq]
        )
        positive_headers = ";".join(
            header for header, _ in positive_map[seq]
        )

        out.write(
            f"{seq}\t{external_headers}\t{positive_headers}\n"
        )

# ============================================================
# SUMMARY
# ============================================================

summary = [
    "FASTA SEQUENCE COMPARISON SUMMARY",
    "=" * 50,
    f"External FASTA: {EXTERNAL_FILE}",
    f"Positive FASTA: {POSITIVE_FILE}",
    "",
    f"External FASTA records: {len(external)}",
    f"Positive FASTA records: {len(positive)}",
    f"Unique external sequences: {len(external_sequences)}",
    f"Unique positive sequences: {len(positive_sequences)}",
    f"Common unique sequences: {len(common)}",
    f"External-only unique sequences: {len(external_only)}",
    f"Positive-only unique sequences: {len(positive_only)}",
    "",
    "DNA/RNA normalization: U converted to T",
    "",
    "Output files:",
    str(COMMON_EXTERNAL),
    str(COMMON_POSITIVE),
    str(EXTERNAL_ONLY),
    str(POSITIVE_ONLY),
    str(COMMON_TABLE),
]

with open(SUMMARY_FILE, "w") as out:
    out.write("\n".join(summary) + "\n")

print("\n".join(summary))
print(f"\nSummary saved to: {SUMMARY_FILE}")
