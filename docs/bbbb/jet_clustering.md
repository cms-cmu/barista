# Unit tests

All:

```bash
 python -m unittest jet_clustering.tests.test_clustering.clusteringTestCase
```

To run one test:

```bash
 python -m unittest jet_clustering.tests.test_clustering.clusteringTestCase.test_synthetic_datasets_bbjjets
```

# To run clustering 

Needed to make the pdf templates

Check the commands in 
`/coffea4bees/scripts//synthetic-dataset-cluster.sh`


# To make clustering PDFs

```bash
python  jet_clustering/make_jet_splitting_PDFs.py analysis/hists/synthetic_datasets_all.coffea  --out jet_clustering/jet-splitting-PDFs-00-05-00/
```


# To create the synthetic datasets 

This takes the clustered PDFs as inputs

Check the commands in 
`/coffea4bees/scripts//synthetic-dataset-decluster.sh`



# To compare synthetic and nominal datasets

```bash
python jet_clustering/compare_datasets.py  --combine_input_files analysis/hists/histData.coffea  analysis/hists/test_synthetic_data_seedXXX.coffea  --out analysis/plots_synthetic_datasets_all_00-07-02-wPSData -m analysis/metadata/plotsSyntheticVsData2.yml
```



# Debugging splittings


Compare splittings:

```python
python  jet_clustering/splitting_comparison_plots.py analysis/hists/test_synthetic_datasets_4j_and_5j.coffea  --out jet_clustering/jet-splitting-PDFs-00-02-00/comparison
```

Check the reclustered splittings (need to run `processor_cluster_4b.py` with `cluster_and_decluster.yml` configuration.
```python
python jet_clustering/check_reclusted_splittings.py analysis/hists/test_synthetic_datasets_4j_and_5j.coffea --out jet_clustering/jet-splitting-PDFs-00-02-00/reclustering
```

# Make pdflatex slides

```bash
awk -f makeslides.awk nominal.config plots_jet_clustering.config > testTexSlides.tex
pdflatex testTexSlide.tex
```
wehre the awk files and the config can be found in the repo: `git@github.com:johnalison/lab.git`
