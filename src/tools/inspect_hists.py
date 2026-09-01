from coffea.util import load
import sys

fname = sys.argv[1] if len(sys.argv) > 1 else "output/ttHbb/histAll_NoJCM.coffea"
h = load(fname)
hists = h.get("hists", h)
print("Histogram names (first 10):", list(hists.keys())[:10])
for k in ["selJets.n", "tagJets.n", "selJets_noJCM.n", "tagJets_noJCM.n"]:
    print(f"Is '{k}' in hists?:", k in hists)

first = list(hists.keys())[0]
print(f"Axes of {first}:", [ax.name for ax in hists[first].axes])
if "process" in [ax.name for ax in hists[first].axes]:
    print("Processes:", [p for p in hists[first].axes["process"]])
