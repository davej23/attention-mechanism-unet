# MSc Data Science Dissertation/Project

U-Net with Attention Mechanism for Detecting Deforestation in Satellite Imagery

## Files
+ **dataset** -- Folder of original dataset from https://zenodo.org/record/3233081
+ **amazon-processed** -- Folder of numpy .npy arrays of processed GeoTIFFs from https://zenodo.org/record/4498086#.YMh3GfKSmCU
+ Workbook.ipynb -- Jupyter notebook of data processing, augmentation, model training and testing.
+ preprocess.py -- Python script to preprocess GeoTIFFs in **amazon-processed**
+ model-evaluate.py -- Python script to print metrics (accuracy, precision, recall, F1-score) for U-Net, AM U-Net which are trained on different data
+ requirements.txt -- Required Python libraries
