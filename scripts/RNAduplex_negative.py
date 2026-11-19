import os
import subprocess
import pandas as pd
import numpy as np
from scipy.stats import ranksums

# 1. Load your positive TSV data
df = pd.read_csv("all_positive_with_sequences.tsv", sep="\t")

# Global filenames for RNAduplex interaction
duplex_filename = "temp_candidates.fasta"
output_filename = "temp_rnaterminal_out.txt"

# All original columns to preserve for both positive and negative entries
keep_cols = [
    "SRR", "SiteID", "Query", "Transcript", "TSlice", "MFEperfect", 
    "MFEratio", "AllenScore", "DegradomeCategory", "DegradomePval", 
    "Tplot_file_path", "tRF_sequence", "Transcript_sequence"
]
final_rows = []

# Loop through each row of the positive dataset
for idx, row in df.iterrows():
    mrna_start = int(row["TStart"])
    mrna_end = int(row["TStop"])
    
    mrna_transcript = str(row["Transcript_sequence"])
    trf_seq = str(row["tRF_sequence"])
    
    before_site = mrna_transcript[:mrna_start]
    after_site = mrna_transcript[mrna_end:]
    binding_site = mrna_transcript[mrna_start:mrna_end]
    
    window_size = mrna_end - mrna_start
    candidate_sites = []
    
    # Track the global transcript start offset index for coordinate reconstruction later
    candidate_offsets = [] 
    
    # Helper function to evaluate candidate windows
    def scan_region(region_seq, global_offset):
        for i in range(len(region_seq) - window_size):
            candidate_neg = region_seq[i : i + window_size]
            
            if candidate_neg.count("CG") == binding_site.count("CG"):
                candidate_g = [1 if char == "G" else 0 for char in candidate_neg]
                binding_g = [1 if char == "G" else 0 for char in binding_site]
                
                stat, p_val = ranksums(candidate_g, binding_g)
                if np.isnan(p_val) or p_val >= 0.05:
                    candidate_sites.append(candidate_neg)
                    candidate_offsets.append(global_offset + i)

    scan_region(before_site, 0)
    scan_region(after_site, mrna_end)
    
    if not candidate_sites:
        continue
        
    # ------------------------------------------------------------------------
    # RUN RNADUPLEX VIA SUBPROCESS
    # ------------------------------------------------------------------------
    with open(duplex_filename, "w") as f:
        for cand in candidate_sites:
            f.write(f">{row['Query']}\n{trf_seq}\n")
            f.write(f">{row['Transcript']}\n{cand}\n")
            
    try:
        with open(duplex_filename, "r") as infile, open(output_filename, "w") as outfile:
            subprocess.run(["RNAduplex"], stdin=infile, stdout=outfile, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error running RNAduplex at row {idx}: {e}")
        continue

    # ------------------------------------------------------------------------
    # PARSE ADVANCED RNADUPLEX DATA METRICS
    # ------------------------------------------------------------------------
    parsed_structures = []
    parsed_mfes = []
    parsed_coords = [] # Holds (q_start, q_end, t_start_local, t_end_local)
    
    with open(output_filename, "r") as f:
        for line in f:
            if not line.startswith(">") and "(" in line and ")" in line:
                # Example line structure: 
                # ((((((((((((((((&.))))))))))).)))).   1,15  : 1081,1098 (-20.30)
                if ":" in line:
                    left_side, right_side = line.split(":")
                    
                    # 1. Extract Dot-Bracket Structure
                    db_struct = line.split()[0]
                    parsed_structures.append(db_struct)
                    
                    # 2. Extract Query Coordinates (last item on the left side of the colon)
                    q_coords_part = left_side.strip().split()[-1]
                    q_start, q_end = map(int, q_coords_part.split(","))
                    
                    # 3. Extract Target Coordinates (first item on the right side of the colon)
                    t_coords_part = right_side.strip().split()[0]
                    t_start_local, t_end_local = map(int, t_coords_part.split(","))
                    
                    parsed_coords.append((q_start, q_end, t_start_local, t_end_local))
                    
                    # 4. Extract MFE value inside parentheses
                    start_idx = line.rfind("(") + 1
                    end_idx = line.rfind(")")
                    try:
                        parsed_mfes.append(float(line[start_idx:end_idx].strip()))
                    except ValueError:
                        parsed_mfes.append(0.0)

    if not parsed_mfes or len(parsed_mfes) != len(candidate_sites):
        continue

    # Find the candidate matching the lowest structural binding minimum
    best_neg_idx = np.argmin(parsed_mfes)
    
    best_neg_seq = candidate_sites[best_neg_idx]
    best_neg_mfe = parsed_mfes[best_neg_idx]
    best_neg_struct = parsed_structures[best_neg_idx]
    
    # Reconstruct absolute global TStart and TStop coordinates relative to the full mRNA
    global_window_start = candidate_offsets[best_neg_idx]
    q_start, q_end, local_start, local_end = parsed_coords[best_neg_idx]
    
    # Handle matching orientation layout boundaries safely
    actual_start_offset = min(local_start, local_end) - 1
    actual_end_offset = max(local_start, local_end)
    
    neg_TStart = global_window_start + actual_start_offset
    neg_TStop = global_window_start + actual_end_offset
    
    # Parse Paired vs Unpaired indicators out of structural dot-brackets
    unpaired_count = best_neg_struct.count(".")
    
    # ------------------------------------------------------------------------
    # FIXED COORD FORMAT FOR PAIRED COLUMN (e.g., "1-15,1081-1098")
    # ------------------------------------------------------------------------
    # Convert 0-based neg_TStart back to 1-based format to match RNAduplex output style
    neg_paired_str = f"{q_start}-{q_end},{neg_TStart + 1}-{neg_TStop}"
    neg_unpaired_str = f"{unpaired_count} bases open loop structural context" 
    
    # ------------------------------------------------------------------------
    # APPEND BALANCED ROWS
    # ------------------------------------------------------------------------
    base_data = {c: row[c] for c in keep_cols}
    
    # Positive Row
    pos_row = base_data.copy()
    pos_row.update({
        "TStart": row["TStart"], "TStop": row["TStop"], "MFEsite": row["MFEsite"],
        "Paired": row["Paired"], "Unpaired": row["Unpaired"], "Structure": row["Structure"],
        "Sequence": row["Sequence"], "Label": 1
    })
    
    # Negative Decoy Row
    neg_row = base_data.copy()
    neg_row.update({
        "TStart": neg_TStart,
        "TStop": neg_TStop,
        "MFEsite": best_neg_mfe,
        "Paired": neg_paired_str,
        "Unpaired": neg_unpaired_str,
        "Structure": best_neg_struct,
        "Sequence": f"{trf_seq}&{best_neg_seq}",
        "Label": 0
    })
    
    final_rows.extend([pos_row, neg_row])

# ------------------------------------------------------------------------
# SAVE OUTPUT MATRIX FILE
# ------------------------------------------------------------------------
for temp_file in [duplex_filename, output_filename]:
    if os.path.exists(temp_file):
        os.remove(temp_file)

if final_rows:
    output_df = pd.DataFrame(final_rows)
    ordered_cols = list(df.columns) + ["Label"]
    output_df = output_df.reindex(columns=ordered_cols)
    output_df.to_csv("all_negative_RNADuplex_with_sequences.tsv", sep="\t", index=False)
    print("Success! Clean 'all_negative_RNADuplex_with_sequences.tsv' generated with all columns populated.")
else:
    print("Process complete. No qualifying decoy sequences passed pipeline thresholds.")
