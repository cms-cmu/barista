#  analysis

To run the analysis, remember first to set the coffea environment and your grid certificate. If you followed the instructions in the [README.md](../../README.md), the `set_shell.sh` file must be located right after the package, and then:

```{bash}
voms-proxy-init -rfc -voms cms --valid 168:00
cd / ## Or go to this directory
source set_shell.sh
```

## In this folder

Here you find the code to run the analysis from `picoAODs` and to make some plots. 
Each folder contains:
 - [helpers](./helpers/): python files with funcions/classes generic to the analyses
 - [metadata](./metadata/): yml files with the metadata for each analysis. In these files you can find input files, datasets, cross sections, etc.  
 - [processors](./processors/): python files with the processors for each analysis.
 - [pytorchModels](./pytorchModels/): training models
 - [jcm_tools](./jcm_tools/): python files to compute JCM weights and to apply them.
 - weights: JCM txt files with weights
 - tests: python scripts for testing the code.
 - hists (optional): if you run the `runner.py` without a name of the output folder, this folder will be created to store the pickle files.

Then, the run-all script is called `runner.py` and it is one directory below (in [coffea4bees/](../../coffea4bees/)). This script will run local or condor depending on the flag used. To learn all the options of the script, just run:
```
# (inside //coffea4bees/)
python runner.py --help
```

## Run Analysis

### Example to run the analysis

For example, to run a processor you can do:
```
#  (inside //coffea4bees/)
python runner.py -t -o test.coffea -d data TTToHadronic TTToSemiLeptonic TTTo2L2Nu  HH4b  -p analysis/processors/processor_example.py -y UL18  -op output/ -c analysis/metadata/example.yml
```

The output file of this process will be `test.coffea` (a coffea output file), which contains many histograms and cutflows. 



## To debug the code

If you want to debug small portions of the code, you can run it interactively in python by using some commands like:
```{python}
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
fname = "root://xrootd-cms.infn.it//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/20UL18JMENano_106X_upgrade2018_realistic_v16_L1v1-v1/230000/9EEE27FD-7337-424F-9D7C-A5427A991D07.root"   #### or any nanoaod file
events = NanoEventsFactory.from_root( fname, schemaclass=NanoAODSchema.v6).events()
```


