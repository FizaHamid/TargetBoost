import io
import math
import os
import re
import subprocess
import pandas as pd

# ==========================================
# 1. FASTA PARSER HELPER
# ==========================================


def load_fasta(fasta_path):
    """Parses a FASTA file into a dictionary mapping sequence IDs to uppercase RNA sequences.

    Trims everything after whitespace in headers to match standard sequence IDs.
    """
    sequences = {}
    current_id = None
    current_seq = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id:
                    sequences[current_id] = "".join(current_seq).upper().replace("T", "U")
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            sequences[current_id] = "".join(current_seq).upper().replace("T", "U")

    return sequences


# ==========================================
# 2. RNADUPLEX EXECUTION & PARSING
# ==========================================


def run_rnaduplex(query_seq, target_seq):
    """Executes RNAduplex via subprocess for a query (tRF) and target (Transcript) pair.

    Parses dot-bracket structure, MFE, target coordinates (TStart, TStop), and query binding range.
    """
    q_rna = str(query_seq).strip().upper().replace("T", "U")
    t_rna = str(target_seq).strip().upper().replace("T", "U")

    # Pass sequences directly separated by newline
    input_str = f"{q_rna}\n{t_rna}\n"

    try:
        process = subprocess.Popen(
            ["RNAduplex"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate(input=input_str)
    except FileNotFoundError:
        raise FileNotFoundError(
            "RNAduplex binary not found in system PATH. Ensure ViennaRNA package is installed."
        )

    struct_line = ""
    for line in stdout.splitlines():
        if "&" in line and any(c in line for c in ["(", ")", "."]):
            struct_line = line.strip()
            break

    if not struct_line:
        return None

    # Example stdout:
    # (((((((((((((((&.))))))))))).)))).   1,15  : 1081,1098 (-20.30)
    parts = struct_line.split()
    structure = parts[0]

    # Extract MFE value inside parentheses: (-20.30) -> -20.30
    mfe_match = re.search(r"\(\s*([-+]?\d*\.\d+|\d+)\s*\)", struct_line)
    mfe = float(mfe_match.group(1)) if mfe_match else 0.0

    # Extract tRF (query) binding site (e.g., "1,15" -> "1-15")
    trf_binding_site = parts[1].replace(",", "-") if len(parts) > 1 else f"1-{len(q_rna)}"

    # Parse target coordinates (e.g., from "1081,1098")
    # RNAduplex target coordinates follow " : <TStart>,<TStop>"
    coord_match = re.search(r":\s*(\d+)\s*,\s*(\d+)", struct_line)
    if coord_match:
        t_start = int(coord_match.group(1))
        t_stop = int(coord_match.group(2))
    else:
        t_start = 1
        t_stop = len(t_rna)

    seq_alignment = f"{q_rna}&{t_rna}"

    return {
        "Structure": structure,
        "MFEsite": mfe,
        "TStart": t_start,
        "TStop": t_stop,
        "tRF_binding_site": trf_binding_site,
        "Sequence": seq_alignment,
    }


# ==========================================
# 3. FEATURE EXTRACTION HELPER FUNCTIONS
# ==========================================


def calculate_gc(seq):
    if not seq or pd.isna(seq):
        return 0.0
    seq = str(seq).upper()
    total_bases = len(seq)
    if total_bases == 0:
        return 0.0
    return ((seq.count("G") + seq.count("C")) / total_bases) * 100


def calculate_weighted_au_flank(window_seq, direction="upstream"):
    if not window_seq:
        return 0.0

    seq = str(window_seq).upper().replace("T", "U")
    ordered_bases = list(reversed(seq)) if direction == "upstream" else list(seq)

    weighted_sum = 0.0
    total_weight = 0.0

    for idx, base in enumerate(ordered_bases):
        distance_i = idx + 1
        weight = math.exp(-0.1 * (distance_i - 1))
        p_i = 1.0 if base in ["A", "U"] else 0.0
        weighted_sum += p_i * weight
        total_weight += weight

    return (weighted_sum / total_weight) if total_weight > 0 else 0.0


def get_dinucleotide_composition(seq):
    dinucleotides = [a + b for a in "ACGU" for b in "ACGU"]
    counts = {d: 0 for d in dinucleotides}
    if not seq:
        return counts

    seq = str(seq).upper().replace("T", "U")
    for i in range(len(seq) - 1):
        dini = seq[i : i + 2]
        if dini in counts:
            counts[dini] += 1

    total = sum(counts.values())
    if total > 0:
        return {d: counts[d] / total for d in dinucleotides}
    return counts


def get_base_composition(seq):
    bases = ["A", "C", "G", "U"]
    counts = {b: 0 for b in bases}
    if not seq:
        return counts

    seq = str(seq).upper().replace("T", "U")
    total = len(seq)
    if total > 0:
        return {b: seq.count(b) / total for b in bases}
    return counts


def parse_trf_binding_boundaries(trf_binding_site_str):
    if not trf_binding_site_str or pd.isna(trf_binding_site_str):
        return None, None
    match = re.search(r"(\d+)\s*[-,\s]\s*(\d+)", str(trf_binding_site_str))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def parse_trf_structure_features(
    trf_struct_part, raw_sequence_field, binding_start, binding_stop
):
    if not trf_struct_part:
        return 0, 0, 0, 0, 0

    num_paired = trf_struct_part.count("(")
    num_unpaired = trf_struct_part.count(".")

    trf_seq_alignment = ""
    if pd.notna(raw_sequence_field) and "&" in str(raw_sequence_field):
        seq_parts = str(raw_sequence_field).split("&")
        if len(seq_parts) == 2:
            trf_seq_alignment = seq_parts[0].upper()

    trf_struct_clean = trf_struct_part.replace(".", " ")
    runs = [len(match.group(0)) for match in re.finditer(r"\(+", trf_struct_clean)]
    longest_stem = max(runs) if runs else 0

    longest_stem_pos = 0
    if longest_stem > 0:
        match_string = "(" * longest_stem
        raw_idx = trf_struct_clean.find(match_string)
        if raw_idx != -1:
            if trf_seq_alignment:
                t_seq_before_stem = trf_seq_alignment[:raw_idx]
                longest_stem_pos = len(t_seq_before_stem.replace("-", "")) + 1
            else:
                longest_stem_pos = raw_idx + 1

    pos1_match = 0
    if binding_start is not None and trf_seq_alignment:
        current_ungapped_pos = 0
        target_alignment_idx = -1
        for idx, char in enumerate(trf_seq_alignment):
            if char != "-":
                current_ungapped_pos += 1
            if current_ungapped_pos == 1:
                target_alignment_idx = idx
                break

        if target_alignment_idx != -1 and target_alignment_idx < len(
            trf_struct_part
        ):
            if binding_start <= 1 <= binding_stop:
                if trf_struct_part[target_alignment_idx] == "(":
                    pos1_match = 1

    return (
        num_paired,
        num_unpaired,
        longest_stem,
        longest_stem_pos,
        pos1_match,
    )


def extract_seed_match_features(
    trf_struct_part, raw_sequence_field, trf_seq, binding_start, binding_stop
):
    feats = {
        "Match_8mer": 0,
        "Match_7mer_m8": 0,
        "Match_7mer_m1": 0,
        "Match_7mer_A1": 0,
        "Match_6mer": 0,
    }

    if (
        not trf_struct_part
        or not raw_sequence_field
        or "&" not in str(raw_sequence_field)
    ):
        return feats

    trf_seq_alignment = str(raw_sequence_field).split("&")[0].upper()

    pos_to_struct_idx = {}
    current_ungapped_pos = 0
    for idx, char in enumerate(trf_seq_alignment):
        if char != "-":
            current_ungapped_pos += 1
            pos_to_struct_idx[current_ungapped_pos] = idx

    def check_position_paired(pos):
        if binding_start is not None and binding_stop is not None:
            if binding_start <= pos <= binding_stop:
                struct_idx = pos_to_struct_idx.get(pos, -1)
                if struct_idx != -1 and struct_idx < len(trf_struct_part):
                    return trf_struct_part[struct_idx] == "("
        return False

    seed_8mer = all(check_position_paired(p) for p in range(1, 9))
    seed_7mer_m8 = all(check_position_paired(p) for p in range(2, 9))
    seed_7mer_m1 = all(check_position_paired(p) for p in range(1, 8))
    seed_6mer = all(check_position_paired(p) for p in range(2, 8))

    trf_has_a1 = trf_seq and trf_seq[0] == "A"
    seed_7mer_A1 = seed_6mer and trf_has_a1

    if seed_8mer:
        feats["Match_8mer"] = 1
    if seed_7mer_m8:
        feats["Match_7mer_m8"] = 1
    if seed_7mer_m1:
        feats["Match_7mer_m1"] = 1
    if seed_7mer_A1:
        feats["Match_7mer_A1"] = 1
    if seed_6mer:
        feats["Match_6mer"] = 1

    return feats


# ==========================================
# 4. MAIN RUNNER PIPELINE
# ==========================================


def run_pipeline(trf_fasta_path, transcript_fasta_path, pairs_tsv_path, output_path):
    print("Loading FASTA files...")
    trf_dict = load_fasta(trf_fasta_path)
    tx_dict = load_fasta(transcript_fasta_path)
    print(
        f"Loaded {len(trf_dict)} tRF sequence(s) and {len(tx_dict)} transcript sequence(s)."
    )

    print(f"Reading pair index file: {pairs_tsv_path}...")
    pairs_df = pd.read_csv(pairs_tsv_path, sep="\t")

    feature_list = []
    print("Running RNAduplex and generating ML feature matrix...")

    for idx, row in pairs_df.iterrows():
        query_id = str(row["Query"]).strip()
        transcript_id = str(row["Transcript"]).strip()

        if query_id not in trf_dict:
            print(
                f"Warning: Query ID '{query_id}' not found in tRF FASTA. Skipping row {idx}..."
            )
            continue
        if transcript_id not in tx_dict:
            print(
                f"Warning: Transcript ID '{transcript_id}' not found in Transcript FASTA. Skipping row {idx}..."
            )
            continue

        trf_seq = trf_dict[query_id]
        full_tx_seq = tx_dict[transcript_id]

        try:
            rnaduplex_out = run_rnaduplex(trf_seq, full_tx_seq)
            if not rnaduplex_out:
                print(
                    f"RNAduplex produced no valid output for pair ({query_id}, {transcript_id}). Skipping..."
                )
                continue

            t_start = rnaduplex_out["TStart"]
            t_stop = rnaduplex_out["TStop"]
            mfe = rnaduplex_out["MFEsite"]
            raw_structure = rnaduplex_out["Structure"]
            seq_field = rnaduplex_out["Sequence"]
            binding_site_str = rnaduplex_out["tRF_binding_site"]

            # Extract Features
            t_start_idx = max(0, t_start - 1)
            tx_length = len(full_tx_seq)
            b_start, b_stop = parse_trf_binding_boundaries(binding_site_str)

            if "&" in raw_structure:
                struct_parts = raw_structure.split("&")
                trf_struct_part = struct_parts[0]
                target_struct_part = struct_parts[1]
            else:
                trf_struct_part = raw_structure
                target_struct_part = ""

            target_site_window_seq = full_tx_seq[t_start_idx:t_stop]

            trf_len = len(trf_seq)
            trf_gc = calculate_gc(trf_seq)
            gc_percent_seed = calculate_gc(target_site_window_seq)
            dist_5prime = t_start_idx
            dist_3prime = tx_length - t_stop

            up_30_start = max(0, t_start_idx - 30)
            up_30_window = full_tx_seq[up_30_start:t_start_idx]
            down_30_end = min(tx_length, t_stop + 30)
            down_30_window = full_tx_seq[t_stop:down_30_end]

            weighted_au_upstream = calculate_weighted_au_flank(
                up_30_window, direction="upstream"
            )
            weighted_au_downstream = calculate_weighted_au_flank(
                down_30_window, direction="downstream"
            )
            combined_weighted_local_au = (
                weighted_au_upstream + weighted_au_downstream
            ) / 2.0

            up_30_dini = get_dinucleotide_composition(up_30_window)
            down_30_dini = get_dinucleotide_composition(down_30_window)
            up_30_base = get_base_composition(up_30_window)
            down_30_base = get_base_composition(down_30_window)

            (
                num_paired,
                num_unpaired,
                longest_stem,
                longest_stem_pos,
                pos1_match,
            ) = parse_trf_structure_features(
                trf_struct_part, seq_field, b_start, b_stop
            )

            pos_1_base = trf_seq[0] if len(trf_seq) > 0 else "N"
            seed_len = t_stop - t_start_idx

            up_2 = (
                full_tx_seq[max(0, t_start_idx - 2) : t_start_idx]
                .zfill(2)
                .replace("0", "N")
            )
            down_2 = full_tx_seq[t_stop : min(tx_length, t_stop + 2)].ljust(2, "N")

            seed_feats = extract_seed_match_features(
                trf_struct_part, seq_field, trf_seq, b_start, b_stop
            )

            record = {
                "SiteID": row.get("SiteID", f"{query_id}:{transcript_id}"),
                "Query_tRF": query_id,
                "TranscriptID": transcript_id,
                "Target_Structure": target_struct_part,
                "tRF_Structure": trf_struct_part,
                "Length_of_tRF": trf_len,
                "GC_Percent_tRF": trf_gc,
                "GC_Percent_Seed": gc_percent_seed,
                "Weighted_Local_AU_Content": combined_weighted_local_au,
                "Distance_to_5_End": dist_5prime,
                "Distance_to_3_End": dist_3prime,
                "Base_Identity_2nt_Upstream_Pos2": up_2[0],
                "Base_Identity_2nt_Upstream_Pos1": up_2[1],
                "Base_Identity_2nt_Downstream_Pos1": down_2[0],
                "Base_Identity_2nt_Downstream_Pos2": down_2[1],
                "Position_1_Match": pos1_match,
                "Position_1": pos_1_base,
                "Target_Length_involved_in_base_Pairing": seed_len,
                "Length_Longest_Consecutive_Pairing": longest_stem,
                "Pos_Longest_Consecutive_Pairing_tRF5": longest_stem_pos,
                "Num_Paired_Positions": num_paired,
                "Num_Unpaired_Positions": num_unpaired,
                "Minimum_Free_Energy": mfe,
                "Match_8mer": seed_feats["Match_8mer"],
                "Match_7mer_m8": seed_feats["Match_7mer_m8"],
                "Match_7mer_m1": seed_feats["Match_7mer_m1"],
                "Match_7mer_A1": seed_feats["Match_7mer_A1"],
                "Match_6mer": seed_feats["Match_6mer"],
            }

            for dini, freq in up_30_dini.items():
                record[f"Upstream_30nt_Dini_{dini}"] = freq
            for dini, freq in down_30_dini.items():
                record[f"Downstream_30nt_Dini_{dini}"] = freq
            for b, freq in up_30_base.items():
                record[f"Upstream_30nt_Base_{b}"] = freq
            for b, freq in down_30_base.items():
                record[f"Downstream_30nt_Base_{b}"] = freq

            feature_list.append(record)

        except Exception as e:
            print(
                f"Skipping index pair ({query_id}, {transcript_id}) due to processing error: {e}"
            )
            continue

    out_df = pd.DataFrame(feature_list)
    out_df.to_csv(output_path, sep="\t", index=False)
    print(f"\nPipeline finished! Feature matrix saved to: {output_path}")
    print(
        f"Matrix Dimensions: {out_df.shape[0]} samples x {out_df.shape[1]} descriptors."
    )


if __name__ == "__main__":
    TRF_FASTA = "trfs.fasta"
    TRANSCRIPT_FASTA = "transcripts.fasta"
    PAIRS_TSV = "pairs.tsv"
    OUTPUT_FILE = "extracted_features_rnaduplex.tsv"

    if (
        os.path.exists(TRF_FASTA)
        and os.path.exists(TRANSCRIPT_FASTA)
        and os.path.exists(PAIRS_TSV)
    ):
        run_pipeline(TRF_FASTA, TRANSCRIPT_FASTA, PAIRS_TSV, OUTPUT_FILE)
    else:
        print(
            "Missing input files! Make sure 'trfs.fasta', 'transcripts.fasta', and 'pairs.tsv' exist."
        )
