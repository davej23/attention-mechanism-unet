# MSc Data Science Dissertation/Project

**An Attention-Based U-Net for Detecting Deforestation Within Satellite Sensor Imagery.**

## Datasets
### Amazon 1 (Regular 3-dim Dataset) -- https://zenodo.org/record/3233081
### Amazon 2 (Larger 4-band Amazon and Atlantic Datasets) -- https://zenodo.org/record/4498086#.YMh3GfKSmCU

## Files
+ **dataset** -- Folder of original dataset from Regular Dataset.
+ **figures** -- Figures for report (amazon-atlantic-forest-mapjpg.jpg from https://pubmed.ncbi.nlm.nih.gov/20433744/).
  + **shapefiles** -- Shapefiles for map. Amazon Shapefile from: (http://worldmap.harvard.edu/data/geonode:amapoly_ivb), rest from: (http://terrabrasilis.dpi.inpe.br/en/download-2/).
+ **metrics** -- Folder of metrics (accuracy, precision, recall, F1-score) for each result.
+ Experimentation.ipynb -- Jupyter notebook of data processing, augmentation, model training and testing.
+ Figures.ipynb -- Jupyter notebook of figures found in **figures**.
+ model-evaluate.py -- Python script to print metrics (accuracy, precision, recall, F1-score) for U-Net, AM U-Net which are trained on different data.
+ predictor.py -- Takes any input RGB image and outputs Attention U-Net-predicted deforestation mask to file.
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
