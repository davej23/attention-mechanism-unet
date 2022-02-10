# MSc Data Science Dissertation/Project

**An Attention-Based U-Net for Detecting Deforestation Within Satellite Sensor Imagery.** 
https://www.sciencedirect.com/science/article/pii/S0303243422000113

## Datasets
### Amazon 1 (Regular 3-dim Dataset) -- https://zenodo.org/record/3233081
### Amazon 2 (Larger 4-band Amazon and Atlantic Datasets) -- https://zenodo.org/record/4498086#.YMh3GfKSmCU

## Files
+ **dataset** -- Folder of original dataset from Regular Dataset.
+ **figures** -- Figures for report (amazon-atlantic-forest-mapjpg.jpg from https://pubmed.ncbi.nlm.nih.gov/20433744/).
  + **shapefiles** -- Shapefiles for map. Amazon Shapefile from: (http://worldmap.harvard.edu/data/geonode:amapoly_ivb), rest from: (http://terrabrasilis.dpi.inpe.br/en/download-2/).
+ **models** -- Folder of each of the three types of Attention U-Net model; load into Keras using 'load_model([modelfilename])'.
+ **metrics** -- Folder of metrics (accuracy, precision, recall, F1-score) for each result.
+ Experimentation.ipynb -- Jupyter notebook of data processing, augmentation, model training and testing.
+ Figures.ipynb -- Jupyter notebook of figures found in **figures**.
+ predictor.py -- Takes any input RGB or 4-band image and outputs Attention U-Net-predicted deforestation mask to file.
+ preprocess-4band-amazon-data.py -- Python script to preprocess GeoTIFFs from 4-band Amazon Dataset and export as numpy pickles.
+ preprocess-4band-atlantic-forest-data.py -- Python script to preprocess GeoTIFFs from 4-band Atlantic Forest Dataset and export as numpy pickles.
+ preprocess-rgb-data.py -- Python script to preprocess data in RGB Dataset and export as numpy pickles.
+ requirements.txt -- Required Python libraries.

## How to use
### Obtaining Attention U-Net Deforestation Masks
+ Run pip -r requirements.txt to install libraries.
+ Download 'unet-attention-3d.hdf5', 'unet-attention-4d.hdf5' and 'unet-attention-4d-atlantic.hdf5' models, and place in same directory as script.
+ Run 'python predictor.py [MODEL IDENTIFIER] [INPUT IMAGE PATH]' or 'python3 predictor.py [MODEL IDENTIFIER] [INPUT IMAGE PATH]'.
  + Model identifier for RGB is 1, 4-band Amazon-trained is 2, 4-band Atlantic Forest-trained is 3.
  + e.g. Get mask prediction of image named 'test.tif' from 4-band Amazon model: 'python predictor.py 2 test.tif'.

### Obtaining Pre-Processed Data
+ Run pip -r requirements.txt to install libraries.
+ Run 'preprocess-4band-amazon-data.py' to pre-process 4-band Amazon data.
+ Run 'preprocess-4band-atlantic-forest-data.py' to pre-process 4-band Atlantic Forest data.
+ Run 'preprocess-rgb-data.py' to pre-process RGB Amazon data.
