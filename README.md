# MSc Data Science Dissertation/Project

U-Net with Attention Mechanism (AM) for Detecting Deforestation in Satellite Imagery

## Datasets
### Amazon 1 (Regular) -- https://zenodo.org/record/3233081
### Amazon 2 (Large) -- https://zenodo.org/record/4498086#.YMh3GfKSmCU

## Files
+ **dataset** -- Folder of original dataset from Regular
+ Workbook.ipynb -- Jupyter notebook of data processing, augmentation, model training and testing.
+ preprocess-large.py -- Python script to preprocess GeoTIFFs from Large
+ preprocess-regular.py -- Python script to preprocess data in Regular
+ model-evaluate.py -- Python script to print metrics (accuracy, precision, recall, F1-score) for U-Net, AM U-Net which are trained on different data
+ requirements.txt -- Required Python libraries

## Releases
+ Models.zip -- Pre-trained models of U-Net and AM U-Net on Regular and Regular (Augmented) data (NOT YET UP TO DATE)
+ amazon-processed-larger.zip -- Folder of numpy .npy arrays of processed GeoTIFFs from Large
+ amazon-processed-regular.zip -- Folder of numpy .npy arrays of processed data from Regular
