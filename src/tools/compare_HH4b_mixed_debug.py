import sys
import os
import coffea.util
import numpy as np

def compare(new_file, ref_mixed_file, ref_data_file):
    print(f"Loading New Debug File: {new_file}")
    new_obj = coffea.util.load(new_file)
    print(f"Loading Ref Mixed File: {ref_mixed_file}")
    ref_m_obj = coffea.util.load(ref_mixed_file)
    print(f"Loading Ref Data File:  {ref_data_file}")
    ref_d_obj = coffea.util.load(ref_data_file)
    
    # Extract cutFlowFourTag
    new_cf = new_obj.get("cutFlowFourTag", {})
    ref_m_cf = ref_m_obj.get("cutFlowFourTag", {})
    ref_d_cf = ref_d_obj.get("cutFlowFourTag", {})
    
    print("\n" + "="*110)
    print(f"{'Cut / Selection':<28} | {'New mix_v0_2018':<18} | {'Ref mix_v0_2018':<18} | {'Diff % (New/Ref)':<18} | {'Ref 2018 Data':<18}")
    print("="*110)
    
    # New debug mix_v0_2018 counts
    new_counts = {}
    for ds, counts in new_cf.items():
        if "mix_v0" in ds or "mixed" in ds:
            for c, val in counts.items():
                new_counts[c] = new_counts.get(c, 0.0) + (val if val is not None else 0.0)
                
    # Ref mix_v0_2018 counts
    ref_m_counts = ref_m_cf.get("mix_v0_2018", {})
    
    # Ref 2018 Data counts (summed over UL18A, B, C, D)
    ref_d_counts = {}
    for ds, counts in ref_d_cf.items():
        if "data_UL18" in ds:
            for c, val in counts.items():
                ref_d_counts[c] = ref_d_counts.get(c, 0.0) + (val if val is not None else 0.0)
                
    all_cuts = sorted(set(list(new_counts.keys()) + list(ref_m_counts.keys()) + list(ref_d_counts.keys())))
    
    for cut in all_cuts:
        n_val = new_counts.get(cut, 0.0)
        rm_val = ref_m_counts.get(cut, 0.0)
        rd_val = ref_d_counts.get(cut, 0.0)
        
        diff_pct = f"{((n_val - rm_val) / rm_val * 100.0):+.2f}%" if rm_val > 0 else "N/A"
        print(f"{cut:<28} | {n_val:<18.1f} | {rm_val:<18.1f} | {diff_pct:<18} | {rd_val:<18.1f}")
        
    print("="*110)
    
    print("\n" + "="*110)
    print("HISTOGRAM INTEGRALS (fourTag SR and SB)")
    print("="*110)
    
    new_hists = new_obj.get("hists", {})
    ref_m_hists = ref_m_obj.get("hists", {})
    ref_d_hists = ref_d_obj.get("hists", {})
    
    for var in ["SvB_MA.ps", "SvB_MA.ps_hh", "canJets.pt", "v4j.mass"]:
        if var in new_hists and var in ref_m_hists:
            h_new = new_hists[var]
            h_ref_m = ref_m_hists[var]
            h_ref_d = ref_d_hists.get(var, None)
            
            # Sum values
            # New: process 'mix_v0'
            h_new_sr = h_new[{'process': 'mix_v0', 'region': 'SR', 'tag': 'fourTag'}].values().sum() if 'SR' in h_new.axes['region'] else 0.0
            h_new_sb = h_new[{'process': 'mix_v0', 'region': 'SB', 'tag': 'fourTag'}].values().sum() if 'SB' in h_new.axes['region'] else 0.0
            
            # Ref mix: process 'mix_v0'
            h_ref_m_sr = h_ref_m[{'process': 'mix_v0', 'region': 'SR', 'tag': 'fourTag'}].values().sum() if 'SR' in h_ref_m.axes['region'] else 0.0
            h_ref_m_sb = h_ref_m[{'process': 'mix_v0', 'region': 'SB', 'tag': 'fourTag'}].values().sum() if 'SB' in h_ref_m.axes['region'] else 0.0
            
            # Ref data: sum across all data processes
            h_ref_d_sr = h_ref_d[{'region': 'SR', 'tag': 'fourTag'}].values().sum() if (h_ref_d is not None and 'SR' in h_ref_d.axes['region']) else 0.0
            h_ref_d_sb = h_ref_d[{'region': 'SB', 'tag': 'fourTag'}].values().sum() if (h_ref_d is not None and 'SB' in h_ref_d.axes['region']) else 0.0
            
            print(f"\nHistogram: '{var}'")
            print(f"  SR: New (on-the-fly) = {h_new_sr:10.1f} | Ref Mixed (v0) = {h_ref_m_sr:10.1f} | Ref Data (2018) = {h_ref_d_sr:10.1f}")
            print(f"  SB: New (on-the-fly) = {h_new_sb:10.1f} | Ref Mixed (v0) = {h_ref_m_sb:10.1f} | Ref Data (2018) = {h_ref_d_sb:10.1f}")

if __name__ == "__main__":
    new_f = sys.argv[1] if len(sys.argv) > 1 else "output/HH4b_mixeddata_debug/histMixedData_debug.coffea"
    ref_m = sys.argv[2] if len(sys.argv) > 2 else "reana_outputs/coffea4bees_20250605_0dc846dc_unblinded_ext_ZZZH/histMixedData.coffea"
    ref_d = sys.argv[3] if len(sys.argv) > 3 else "reana_outputs/coffea4bees_20250605_0dc846dc_unblinded_ext_ZZZH/histAll.coffea"
    compare(new_f, ref_m, ref_d)
