import sys
import os
import coffea.util

def compare_files(data_file, mixed_file):
    print("Loading Data:", data_file)
    d_obj = coffea.util.load(data_file)
    print("Loading MixedData:", mixed_file)
    m_obj = coffea.util.load(mixed_file)
    
    d_cf = d_obj.get("cutFlowFourTag", {})
    m_cf = m_obj.get("cutFlowFourTag", {})
    
    print("\n=== FourTag CutFlow Comparison ===")
    data_totals = {}
    for ds, counts in d_cf.items():
        if "data" in ds:
            for cut, val in counts.items():
                data_totals[cut] = data_totals.get(cut, 0.0) + (val if val is not None else 0.0)
                
    mixed_totals = {}
    n_samples = 0
    for ds, counts in m_cf.items():
        if "mix" in ds:
            n_samples += 1
            for cut, val in counts.items():
                mixed_totals[cut] = mixed_totals.get(cut, 0.0) + (val if val is not None else 0.0)
                
    scale_factor = 1.0 / n_samples if n_samples > 0 else 1.0
    
    header = "{:<30} | {:<15} | {:<20} | {:<18}".format("Cut / Selection", "Data 4b Yield", "MixedData 4b (avg)", "Ratio (Data/Mixed)")
    print(header)
    print("-" * len(header))
    
    all_cuts = sorted(set(list(data_totals.keys()) + list(mixed_totals.keys())))
    for cut in all_cuts:
        d_val = data_totals.get(cut, 0.0)
        m_val = mixed_totals.get(cut, 0.0) * scale_factor
        ratio_str = "{:.4f}".format(d_val / m_val) if m_val > 0 else "N/A"
        print("{:<30} | {:<15.1f} | {:<20.1f} | {:<18}".format(cut, d_val, m_val, ratio_str))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/tools/compare_fourtag_data_mixeddata.py <data.coffea> <mixeddata.coffea>")
        sys.exit(1)
    compare_files(sys.argv[1], sys.argv[2])
