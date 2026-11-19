import io
import os
import re
import math
import pandas as pd

def calculate_gc(seq):
    """Calculates GC percentage of a sequence string."""
    if not seq or pd.isna(seq): return 0.0
    seq = str(seq).upper()
    total_bases = len(seq)
    if total_bases == 0: return 0.0
    return ((seq.count('G') + seq.count('C')) / total_bases) * 100

def calculate_weighted_au_flank(window_seq, direction='upstream'):
    """
    Implements the proximity-weighted AU content formula:
    Sum( P(i) * exp(-0.1 * (i-1)) ) / Sum( exp(-0.1 * (i-1)) )
    """
    if not window_seq:
        return 0.0
    
    seq = str(window_seq).upper().replace('T', 'U')
    
    if direction == 'upstream':
        ordered_bases = list(reversed(seq))
    else:
        ordered_bases = list(seq)
        
    weighted_sum = 0.0
    total_weight = 0.0
    
    for idx, base in enumerate(ordered_bases):
        distance_i = idx + 1
        weight = math.exp(-0.1 * (distance_i - 1))
        p_i = 1.0 if base in ['A', 'U'] else 0.0
        weighted_sum += p_i * weight
        total_weight += weight
        
    return (weighted_sum / total_weight) if total_weight > 0 else 0.0

def get_dinucleotide_composition(seq):
    """Returns frequency dictionary of all 16 dinucleotides in a sequence window."""
    dinucleotides = [a+b for a in "ACGU" for b in "ACGU"]
    counts = {d: 0 for d in dinucleotides}
    if not seq: return counts
    
    seq = str(seq).upper().replace('T', 'U')
    for i in range(len(seq) - 1):
        dini = seq[i:i+2]
        if dini in counts:
            counts[dini] += 1
            
    total = sum(counts.values())
    if total > 0:
        return {d: counts[d] / total for d in dinucleotides}
    return counts

def get_base_composition(seq):
    """Returns absolute frequency ratios of individual A, C, G, U bases in a window."""
    bases = ["A", "C", "G", "U"]
    counts = {b: 0 for b in bases}
    if not seq: return counts
    
    seq = str(seq).upper().replace('T', 'U')
    total = len(seq)
    if total > 0:
        return {b: seq.count(b) / total for b in bases}
    return counts

def parse_trf_binding_boundaries(trf_binding_site_str):
    """Parses a string like '1-14' into 1-based start and stop integers."""
    if not trf_binding_site_str or pd.isna(trf_binding_site_str):
        return None, None
    match = re.search(r'(\d+)\s*-\s*(\d+)', str(trf_binding_site_str))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def parse_trf_structure_features(trf_struct_part, raw_sequence_field, binding_start, binding_stop):
    """
    Parses structural pairing metrics using alignment string logic and explicit binding boundaries.
    """
    if not trf_struct_part:
        return 0, 0, 0, 0, 0
        
    num_paired = trf_struct_part.count(')')
    num_unpaired = trf_struct_part.count('.')
    
    right_seq = ""
    if pd.notna(raw_sequence_field) and '&' in str(raw_sequence_field):
        seq_parts = str(raw_sequence_field).split('&')
        if len(seq_parts) == 2:
            right_seq = seq_parts[1].upper()
            
    # Find longest consecutive segment of closing brackets ')'
    trf_struct_clean = trf_struct_part.replace('.', ' ')
    runs = [len(match.group(0)) for match in re.finditer(r'\)+', trf_struct_clean)]
    longest_stem = max(runs) if runs else 0
    
    longest_stem_pos = 0
    if longest_stem > 0:
        match_string = ')' * longest_stem
        raw_idx = trf_struct_clean.find(match_string)
        if raw_idx != -1:
            if right_seq:
                # Map out alignment gaps to find true structural position on the un-gapped sequence
                t_seq_before_stem = right_seq[:raw_idx]
                longest_stem_pos = len(t_seq_before_stem.replace('-', '')) + 1
            else:
                longest_stem_pos = raw_idx + 1

    # Check Position_1_Match using binding site rules
    pos1_match = 0
    if binding_start is not None and right_seq:
        # Trace through the alignment string (right_seq) to map un-gapped index 1
        current_ungapped_pos = 0
        target_alignment_idx = -1
        for idx, char in enumerate(right_seq):
            if char != '-':
                current_ungapped_pos += 1
            if current_ungapped_pos == 1:
                target_alignment_idx = idx
                break
        
        # Verify if position 1 falls within the active structure map and is paired
        if target_alignment_idx != -1 and target_alignment_idx < len(trf_struct_part):
            if binding_start <= 1 <= binding_stop:
                if trf_struct_part[target_alignment_idx] == ')':
                    pos1_match = 1
                    
    return num_paired, num_unpaired, longest_stem, longest_stem_pos, pos1_match

def extract_seed_match_features(trf_struct_part, raw_sequence_field, trf_seq, binding_start, binding_stop):
    """
    Evaluates tRF structural seed match layouts (8mer, 7mer-m8, 7mer-m1, 7mer-A1, 6mer)
    by mapping un-gapped mature tRF coordinates back into the structural alignment map.
    """
    feats = {"Match_8mer": 0, "Match_7mer_m8": 0, "Match_7mer_m1": 0, "Match_7mer_A1": 0, "Match_6mer": 0}
    
    if not trf_struct_part or not raw_sequence_field or '&' not in str(raw_sequence_field):
        return feats
        
    right_seq = str(raw_sequence_field).split('&')[1].upper()
    
    # Map each 1-based un-gapped position of the tRF sequence to its index in the structural string
    pos_to_struct_idx = {}
    current_ungapped_pos = 0
    for idx, char in enumerate(right_seq):
        if char != '-':
            current_ungapped_pos += 1
            pos_to_struct_idx[current_ungapped_pos] = idx

    def check_position_paired(pos):
        # Checks if an absolute 1-based tRF base position falls in bounds and is paired
        if binding_start <= pos <= binding_stop:
            struct_idx = pos_to_struct_idx.get(pos, -1)
            if struct_idx != -1 and struct_idx < len(trf_struct_part):
                return trf_struct_part[struct_idx] == ')'
        return False

    # Seed checking configurations (Strict 1-based definitions)
    seed_8mer = all(check_position_paired(p) for p in range(1, 9))      # Strictly positions 1-8
    seed_7mer_m8 = all(check_position_paired(p) for p in range(2, 9))   # Strictly positions 2-8
    seed_7mer_m1 = all(check_position_paired(p) for p in range(1, 8))   # Strictly positions 1-7
    seed_6mer = all(check_position_paired(p) for p in range(2, 8))      # Strictly positions 2-7
    
    # A1 requires an Adenine at position 1 of the mature tRF sequence
    trf_has_a1 = (trf_seq and trf_seq[0] == 'A')
    seed_7mer_A1 = seed_6mer and trf_has_a1

    if seed_8mer: feats["Match_8mer"] = 1
    if seed_7mer_m8: feats["Match_7mer_m8"] = 1
    if seed_7mer_m1: feats["Match_7mer_m1"] = 1
    if seed_7mer_A1: feats["Match_7mer_A1"] = 1
    if seed_6mer: feats["Match_6mer"] = 1

    return feats

def process_tsv_features(input_path, output_path):
    print(f"Reading input file: {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    if not lines:
        print("Error: Input file is empty.")
        return
        
    header_line = lines[0]
    if "Transcript_sequenceSRR" in header_line:
        header_line = header_line.replace("Transcript_sequenceSRR", "Transcript_sequence\nSRR")
        lines[0] = header_line
        
    cleaned_data = "".join(lines)
    df = pd.read_csv(io.StringIO(cleaned_data), sep="\t")
    
    feature_list = []
    
    print("Generating updated ML matrix profile and structural parameters...")
    for idx, row in df.iterrows():
        try:
            t_start = int(row['TStart'])
            t_stop = int(row['TStop'])
            
            t_start_idx = t_start - 1
            
            full_tx_seq = str(row['Transcript_sequence']).upper().replace('T', 'U')
            tx_length = len(full_tx_seq)
            
            trf_seq = str(row['tRF_sequence']).upper().replace('T', 'U')
            
            # Unpack structural boundaries
            b_start, b_stop = parse_trf_binding_boundaries(row.get('tRF_binding_site', ''))
            
            # --- Extract Exact Dot-Bracket Structural Components ---
            raw_structure = str(row['Structure'])
            if '&' in raw_structure:
                struct_parts = raw_structure.split('&')
                target_struct_part = struct_parts[0]
                trf_struct_part = struct_parts[1]
            else:
                target_struct_part = raw_structure
                trf_struct_part = ""
            
            target_site_window_seq = full_tx_seq[t_start_idx:t_stop]
            
            # --- Group 1: Intrinsic Attributes ---
            trf_len = len(trf_seq)
            trf_gc = calculate_gc(trf_seq)
            gc_percent_seed = calculate_gc(target_site_window_seq)
            
            # --- Group 2: Spatial Layout Distances ---
            dist_5prime = t_start_idx
            dist_3prime = tx_length - t_stop
            
            # --- Group 3: Context Windows Slicing ---
            up_30_start = max(0, t_start_idx - 30)
            up_30_window = full_tx_seq[up_30_start : t_start_idx]
            
            down_30_end = min(tx_length, t_stop + 30)
            down_30_window = full_tx_seq[t_stop : down_30_end]
            
            # --- Group 4: Local Weighted Proximity AU ---
            weighted_au_upstream = calculate_weighted_au_flank(up_30_window, direction='upstream')
            weighted_au_downstream = calculate_weighted_au_flank(down_30_window, direction='downstream')
            combined_weighted_local_au = (weighted_au_upstream + weighted_au_downstream) / 2.0
            
            # --- Group 5: Dinucleotide and Base Breakdowns ---
            up_30_dini = get_dinucleotide_composition(up_30_window)
            down_30_dini = get_dinucleotide_composition(down_30_window)
            
            up_30_base = get_base_composition(up_30_window)
            down_30_base = get_base_composition(down_30_window)
            
            # --- Group 6: Pairing Metrics & Positions ---
            num_paired, num_unpaired, longest_stem, longest_stem_pos, pos1_match = parse_trf_structure_features(
                trf_struct_part, row['Sequence'], b_start, b_stop
            )
            
            pos_1_base = trf_seq[0] if len(trf_seq) > 0 else "N"
            seed_len = t_stop - t_start_idx
            
            up_2 = full_tx_seq[max(0, t_start_idx-2):t_start_idx].zfill(2).replace('0', 'N')
            down_2 = full_tx_seq[t_stop:min(tx_length, t_stop+2)].ljust(2, 'N')
            
            base_up_pos2 = up_2[0]   
            base_up_pos1 = up_2[1]   
            base_down_pos1 = down_2[0] 
            base_down_pos2 = down_2[1] 
            
            mfe = row['MFEsite']
            
            # --- Group 7: Extract Structural Seed Match Features ---
            seed_feats = extract_seed_match_features(trf_struct_part, row['Sequence'], trf_seq, b_start, b_stop)
            
            record = {
                "SiteID": row.get('SiteID', idx),
                "Query_tRF": row.get('Query', ''),
                "TranscriptID": row.get('Transcript', ''),
                "Target_Structure": target_struct_part,
                "tRF_Structure": trf_struct_part,
                "Length_of_tRF": trf_len,
                "GC_Percent_tRF": trf_gc,
                "GC_Percent_Seed": gc_percent_seed,
                "Weighted_Local_AU_Content": combined_weighted_local_au,
                "Distance_to_5_End": dist_5prime,
                "Distance_to_3_End": dist_3prime,
                "Base_Identity_2nt_Upstream_Pos2": base_up_pos2,
                "Base_Identity_2nt_Upstream_Pos1": base_up_pos1,
                "Base_Identity_2nt_Downstream_Pos1": base_down_pos1,
                "Base_Identity_2nt_Downstream_Pos2": base_down_pos2,
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
                "Match_6mer": seed_feats["Match_6mer"]
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
            print(f"Skipping line index {idx} due to feature generation anomaly: {e}")
            continue
            
    out_df = pd.DataFrame(feature_list)
    out_df.to_csv(output_path, sep="\t", index=False)
    print(f"\nProcessing successful! Cleaned feature output generated at: {output_path}")
    print(f"Matrix Dimensions: {out_df.shape[0]} samples x {out_df.shape[1]} descriptors.")

if __name__ == "__main__":
    INPUT_FILE = "filtered_true.tsv"
    OUTPUT_FILE = "positive_feature_table.tsv"
    
    if os.path.exists(INPUT_FILE):
        process_tsv_features(INPUT_FILE, OUTPUT_FILE)
    else:
        print(f"Missing resource: Place your file under name '{INPUT_FILE}' inside this directory path.")
