#  Plots

## To produce some plots

Assuming that the file with your histograms is called `output/hists.coffea`, you can run:
```
python plots/makePlots.py output/hists.coffea  -o testPlotsNew 

```

### To produce some plots interactively

```
python -i plots/iPlot.py  output/hists.coffea  -o testPlotsNew
```

### Examples

```
>>> examples()
```

### 1D Examples

```
>>> plot("SvB_MA.ps_hh",doRatio=1, debug=True, region="SR",cut="passPreSel",rebin=1,rlim=[0,2],norm=1)
>>> plot("SvB_MA_ps_zh",cut="passPreSel",region="SB",doRatio=True,debug=True,ylabel="Entries",norm=False,legend=True,rebin=5,yscale='log')
```

### 2D Examples

```
>>> plot2d("quadJet_min_dr.lead_vs_subl_m",process="TTToHadronic",region=sum,cut="passPreSel")
>>> plot2d("quadJet_min_dr.lead_vs_subl_m",process="TTToHadronic",region=sum,cut="passPreSel",full=3)
```

### To plot the same process from two different cuts

```
>>> plot("canJet0.pt", region="SR", cut=["passSvB","failSvB"],process="data")
>>> plot("canJet0.pt", region=["SB","SR"],cut="passSvB",process="data")

```

### To plot different processes 

```
>>> plot("v4j.mass", region="SR", cut="passPreSel",process="data",norm=1)
>>> plot("v4j.mass", region="SR", cut="passPreSel",process=["TTTo2L2Nu","data"],norm=1)

```


### To plot the same process from two different inputs

```
> py  -i plots/iPlot.py hists/histAll_file1.coffea hists/histAll_file1.coffea -l file1 file2
```

```
>>> plot("canJet0.pt",region="SR",cut="passPreSel",process="data")
```


