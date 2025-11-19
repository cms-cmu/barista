# Backend

The monitor system consists of a central server that handles the aggregation and visualization of the data and multiple clients that send log, progress and other information to the server. For now, the visualization is limited to the console output provided by [`rich`](https://rich.readthedocs.io/en/stable/index.html) package.

This section of the documentation explains how the classifier runs in the backend to implement it for different physics analyses. The base code of the classifier inside `src/classifier` does not require changes for specific analyses and thus will not be covered here. Most of the examples will refer to the bbreww framework for the HH-bbWW analysis, but these also apply to the coffea4bees framework. 

We run the classifier using one of the run.sh files in classifier/config/workflows/.../run.sh. Most of the variables defined in this file are explained in [HCR training](hcr.md). 

Depending on the analysis, the `CLASSIFIER_CONFIG_PATHS` variable needs to be different. It points to the folder inside which the framework is going to look for modules. All the modules that are common to all analyses are in src/classifier, while analysis specific ones are inside `CLASSIFIER_CONFIG_PATHS/classifier/...`. If both src and the specific path contain the same folder, modules from both folders are still loaded together. Note that the framework looks for specific modules only inside specific folders, so do not change the names of the folders.

## Loading and labeling datasets

Since we are training the classifier with Monte Carlo datasets, the label of the datasets acts as the true values for the classifier training. For example, if we want to train a model to distinguish TTbar Monte Carlo vs. Signal Monte Carlo, we need to tell the classifier which events truly belong to TTbar and Signal. 

To do so, we can create a class that looks for certain patterns inside the datasets metadata file we provide in `.../workflows/.../train.yml`. For example, see `bbreww/classifier/config/dataset/bbWW/_picoAOD`. In this file, we create a `filelists` variable that contains the list of all samples we are going to train on along with a label for each sample. Then we can use this label to create a label (using add_label_index) that the classifier can train on. See `bbreww/classifier.config/dataset/bbWW/svb` for example:

```python
minor_bkgs =  ["WplusJets", "tW", "singleTop"]
        for bkg in minor_bkgs:
            if bkg in self.mc_processes:
                ps.append(
                    _group.fullmatch(
                        (f"label:{bkg}",),
                        processors=[
                            lambda: _signal_selection,
                            lambda: add_label_index("other"),
                        ],
                        name="minor background selection",
                    ),
                )
```

This file also has a dataset normalization implemented, where we can normalize one dataset to have N times the weight of another (using --norm). This --norm parameter is passed in the train.yml file: 

```python
- module: bbWW.svb.Background
    option:
      - --mc-processes ttbar
      - --metadata bbreww/metadata/classifier/datasets_v4
      - --norm 4 #set TTbar to be 4 times weight of signal
      - --friends "" bbreww/metadata/classifier/output_friend.json@@classifier_input # set input friend tree
```

## Input Tensors

The input tensors are loaded from root files created from the processor of the analysis (or any other way outside of the classifier framework). In `.../config/workflows/.../train.yml` a .json file with the path to all these root files and the name of the branches inside them needs to be passed as `--friends` parameter. Then these inputs and any branches they may contain need to be properly configured in `.../config/setting/(input_file).py`. In bbreww, this is bbWWHCR.py. 

## Model

The model itself is stores in `.../classifier/nn/blocks`. The whole framework, including the model is based on PyTorch and thus some physics functions aren't easily available. Some of these functions (like four vector additions or delta_R calculation) are implemented in both `coffea4bees` and `bbreww` model files. 

The user is free to change the model according to the analysis requirements, but the training will only run if the first output of the model has the same dimensions as the number of trainable labels. Let's consider the example of this return line from `bbWW_models.py`: 

```python
return HH_logits, TT_final
```

There are 3 trainable labels in this case, and thus HH_logits here is configure to have shape (n_events, 3) while the second output can have any number of features in the second dimension. The second output returned actually has shape (n_events, 8) and is treated as auxiliary information to improve loss training. 

## Model Training

The main code for the training and evaluation of the model (including inputs) is inside `...classifier/ml/models/(model_file).py`. This file defines input tensors and variables to the model, puts them together, and calls the model function to train and evaluate. One can configure which variables to use in the backpropagation by passing them to `batch[...]`. Take `.../ml/models/bbWWHCR` for example. Here, hh is passed to the training as `batch[Output.hh_raw]`, which makes it a part of the loss training:

```python       
hh, tt = self._nn(*_HCRInput(batch, self._device))
batch[Output.hh_raw] = hh
batch[Output.tt_raw] = tt
        
loss = self._loss(batch)
```

One can also define which variables to save in the evaluation step here. The inputs passed here must match what's defined in `...config/setting/(input_config_file).py` (note that Output is also defined in this same file). 


## Saving Outputs and Evaluation

Now which labels to train with, how to select events, and which outputs to generate as the model trains is defined in `.../config/model/bbWW/HCR/baseline_svb.py` for example. In this example file, a few ROC curves to save are defined with the corresponding trained labels scores to populate the curves with. And which variables to save are defined in the Eval class. Any variables added to batch inside the `model/.../...py` file can be saved to the evaluation output.

Optional: the ROC curves creation and selection in `src/classifier/config/analysis/HCR/_loss_roc.py` should be general to all analyses, but if you want to make specifc changes to this file, create a file inside your analysis repo's `classifier/config/...` e.g. `.../bbWW/_loss_roc.py` and pass this to the analyze step inside run.sh instead of HCR.loss_roc. This step is required if you want to save more information not already included in the defaul loss_roc.py file. 

