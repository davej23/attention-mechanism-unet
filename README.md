# MSc Data Science Dissertation/Project

original title: An attention-based U-Net for detecting Amazon deforestation from satellite imagery.
provisional title: An attention-based U-Net for detecting deforestation within satellite sensor imagery.

## Datasets
### Amazon 1 (Regular 3-dim Dataset) -- https://zenodo.org/record/3233081
### Amazon 2 (Larger 4-band Amazon and Atlantic Datasets) -- https://zenodo.org/record/4498086#.YMh3GfKSmCU

## Files
+ **dataset** -- Folder of original dataset from Regular Dataset.
+ **figures** -- Figures for report (amazon-atlantic-forest-mapjpg.jpg from https://pubmed.ncbi.nlm.nih.gov/20433744/).
+ **metrics** -- Folder of metrics (accuracy, precision, recall, F1-score) for each result.
+ Experimentation.ipynb -- Jupyter notebook of data processing, augmentation, model training and testing.
+ Figures.ipynb -- Jupyter notebook of figures found in **figures**.
+ model-evaluate.py -- Python script to print metrics (accuracy, precision, recall, F1-score) for U-Net, AM U-Net which are trained on different data.
+ preprocess-large.py -- Python script to preprocess GeoTIFFs from Large Dataset.
+ preprocess-regular.py -- Python script to preprocess data in Regular Dataset.
+ requirements.txt -- Required Python libraries.

## How to use
+ Run pip -r requirements.txt to install libraries.
+ Download and extract models from Models.zip in Releases.
+ Download and extract Regular and Large datasets into current working directory from above links.
+ Download and extract pre-processed data from .zip files in Releases
  + **OR** Run preprocess-large.py and pre-process-regular.py scripts.
+ Run model-evaluate.py.

## Releases
+ U-Net Models.zip -- Pre-trained models of U-Net and Attention U-Net on augmented regular 3-dim data and larger 4-dim datasets, with history for each of training and validation losses.
+ ResNet50-SegNet-Regular-Model.zip -- Pre-trained model of ResNet50-SegNet on augmented regular 3-dim data, with history of training and validation losses.
+ ResNet50-SegNet-4-band-Models.zip -- Pre-trained model of ResNet50-SegNet on larger 4-dim datasets, with history for each of training and validation losses.
+ FCN32-VGG16-Models.zip -- Pre-trained model of FCN32-VGG16 on larger 4-dim datasets, with history for each of training and validation losses.
+ amazon-processed.zip -- Contains folders of numpy .npy arrays of processed Large and Regular datasets.
