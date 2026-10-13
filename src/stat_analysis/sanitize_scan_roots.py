#!/usr/bin/env python3
"""
Sanitizes merged MultiDimFit likelihood scan ROOT files for plot1DScan.py.
In Combine, numerical precision around minimum points can yield deltaNLL
values like -1e-9. ROOT's TGraph::Sort() sorts equal X by Y ascending,
causing RemoveGraphXDuplicates() to drop the exact 0.0 point and retain the
slightly negative point. Because plot1DScan.py checks `if graph.GetY()[i] == 0.0:`,
it fails to identify the best fit point and crashes with AssertionError.
This script sets any deltaNLL with abs(deltaNLL) < 1e-5 to exactly 0.0.
"""

import sys
import os
import array

def sanitize_file(fpath):
    if not os.path.exists(fpath):
        return
    try:
        import ROOT
    except ImportError:
        return
    f_in = ROOT.TFile.Open(fpath)
    if not f_in or f_in.IsZombie():
        return
    t_in = f_in.Get("limit")
    if not t_in:
        f_in.Close()
        return

    needs_fix = False
    for i in range(t_in.GetEntries()):
        t_in.GetEntry(i)
        if abs(t_in.deltaNLL) < 1e-5 and t_in.deltaNLL != 0.0:
            needs_fix = True
            break

    if not needs_fix:
        f_in.Close()
        return

    tmp_path = fpath + ".sanitize.tmp"
    f_out = ROOT.TFile.Open(tmp_path, "RECREATE")
    t_out = t_in.CloneTree(0)
    dnll_buf = array.array("f", [0.0])
    t_out.SetBranchAddress("deltaNLL", dnll_buf)

    for i in range(t_in.GetEntries()):
        t_in.GetEntry(i)
        val = t_in.deltaNLL
        if abs(val) < 1e-5:
            val = 0.0
        dnll_buf[0] = val
        t_out.Fill()

    f_out.Write()
    f_out.Close()
    f_in.Close()
    os.replace(tmp_path, fpath)
    print(f"[sanitize_scan_roots] Fixed near-zero deltaNLL values in {fpath}")

def main():
    for f in sys.argv[1:]:
        sanitize_file(f)

if __name__ == "__main__":
    main()
