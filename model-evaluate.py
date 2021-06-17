import tensorflow as tf
import numpy as np
from tensorflow.keras import *
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt 
import os
import PIL
import rioxarray as rxr
from sklearn.metrics import *

base_dir = r"./amazon-processed-regular/"

# Ingest Regular images
print('Ingesting regular dataset...')

## Training images
training_images_list = os.listdir(r"{}training/images/".format(base_dir))
training_masks_list = []
training_images = []
for n in training_images_list:
  training_images.append(np.load(r"{}training/images/{}".format(base_dir,n), allow_pickle=True))
  training_masks_list.append(n[:-9]+'.png')

## Training masks
training_masks = []
for n in training_masks_list:
  training_masks.append(np.load(r"{}training/masks/{}".format(base_dir,n+'.npy'), allow_pickle=True))

## Test images
test_images_list = os.listdir(r"{}test/images".format(base_dir))
test_images = []
for n in test_images_list:
  test_images.append(np.load(r"{}test/images/{}".format(base_dir,n)))

## Validation images
validation_images_list = os.listdir(r"{}validation/images/".format(base_dir))
validation_masks_list = []
validation_images = []
for n in validation_images_list:
  validation_images.append(np.load(r"{}validation/images/{}".format(base_dir,n), allow_pickle=True))
  validation_masks_list.append(n[:-9]+'.png')

## Validation masks
validation_masks = []
for n in validation_masks_list:
  validation_masks.append(np.load(r"{}validation/masks/{}".format(base_dir,n+'.npy'), allow_pickle=True))


print('Ingesting large dataset...')

base_dir2 = r"./amazon-processed-large/"

# Ingest Large images

## Training images
training_images_list2 = os.listdir(r"{}training/images/".format(base_dir2))
training_masks_list2 = []
training_images2 = []
for n in training_images_list2:
  training_masks_list2.append(n)
  training_images2.append(np.load(r"{}training/images/{}".format(base_dir2,n), allow_pickle=True))

## Training masks
training_masks2 = []
for n in training_masks_list2:
  training_masks2.append(np.load(r"{}training/masks/{}".format(base_dir2,n), allow_pickle=True))

## Test images
test_images_list2 = os.listdir(r"{}test/images/".format(base_dir2))
test_masks_list2 = []
test_images2 = []
for n in test_images_list2:
  test_masks_list2.append(n)
  test_images2.append(np.load(r"{}test/images/{}".format(base_dir2,n), allow_pickle=True))

## Test masks
test_masks2 = []
for n in test_masks_list2:
  test_masks2.append(np.load(r"{}test/masks/{}".format(base_dir2,n), allow_pickle=True))

## Validation images
validation_images_list2 = os.listdir(r"{}validation/images/".format(base_dir2))
validation_masks_list2 = []
validation_images2 = []
for n in validation_images_list2:
  validation_masks_list2.append(n)
  validation_images2.append(np.load(r"{}validation/images/{}".format(base_dir2,n), allow_pickle=True))

## Validation masks
validation_masks2 = []
for n in validation_masks_list2:
  validation_masks2.append(np.load(r"{}validation/masks/{}".format(base_dir2,n), allow_pickle=True))


def score_eval(model, image, mask): # Gives score of mask vs prediction
  if type(image) != list:   
    reconstruction = model.predict(image).reshape(mask.shape[1], mask.shape[2])
    reconstruction = np.round(reconstruction).flatten()

    return accuracy_score(mask.flatten(), reconstruction)

  else: # If a list of images input, find accuracy for each
    scores = []
    for i in range(len(image)):
      reconstruction = model.predict(image[i]).reshape(mask[i].shape[1], mask[i].shape[2])
      reconstruction = np.round(reconstruction).flatten()

      scores.append(accuracy_score(mask[i].flatten(), reconstruction))

    return scores

def recall_eval(model, image, mask): # Find recall score
  if type(image) != list:   
    reconstruction = model.predict(image).reshape(mask.shape[1], mask.shape[2])
    reconstruction = np.round(reconstruction).flatten()

    return recall_score(mask.flatten(), reconstruction)

  else: # If a list of images input, find accuracy for each
    recall = []
    for i in range(len(image)):
        reconstruction = model.predict(image[i]).reshape(mask[i].shape[1], mask[i].shape[2])
        reconstruction = np.round(reconstruction).flatten()

        recall.append(recall_score(mask[i].flatten(), reconstruction))

    return recall

def precision_eval(model, image, mask): # Find precision score
  if type(image) != list:   
    reconstruction = model.predict(image).reshape(mask.shape[1], mask.shape[2])
    reconstruction = np.round(reconstruction).flatten()

    return precision_score(mask.flatten(), reconstruction)

  else: # If a list of images input, find accuracy for each
    precision = []
    for i in range(len(image)):
        reconstruction = model.predict(image[i]).reshape(mask[i].shape[1], mask[i].shape[2])
        reconstruction = np.round(reconstruction).flatten()

        precision.append(precision_score(mask[i].flatten(), reconstruction))

    return precision

def f1_score_eval(model, image, mask): # Find F1-score
    prec = np.mean(precision_eval(model, image, mask))
    rec = np.mean(recall_eval(model, image, mask))

    if prec + rec == 0:
        return 0

    return 2 * (prec * rec) / (prec + rec)

def f1_score_eval_basic(precision, recall):
    prec = np.mean(precision)
    rec = np.mean(recall)

    if prec + rec == 0:
        return 0

    return 2 * (prec * rec) / (prec + rec)

def produce_mask(image): # Outputs rounded image (binary)
  return np.round(image)

print('Loading in models...')
am_unet = load_model('./Models/unet-am-regular-data.hdf5')
am_unet_1 = load_model('./Models/unet-am-augmented-data.hdf5')
unet = load_model('./Models/unet-regular-data.hdf5')
unet_1 = load_model('./Models/unet-augmented-data.hdf5')
unet_large = load_model('./Models/unet-large-data.hdf5')
am_unet_large = load_model('./Models/unet-am-large-data.hdf5')

# Scores of each model
print('Calculating scores...')
unet_score = (score_eval(unet, validation_images, validation_masks))
unet1_score = (score_eval(unet_1, validation_images, validation_masks))
am_unet_score = (score_eval(am_unet, validation_images, validation_masks))
am_unet1_score = (score_eval(am_unet_1, validation_images, validation_masks))
unet_large_score = (score_eval(unet_large, validation_images2, validation_masks2))
am_unet_large_score = (score_eval(am_unet_large, validation_images2, validation_masks2))

# Precision and recall of each model
print('Calculating precision statistics...')
unet_precision = (precision_eval(unet, validation_images, validation_masks))
unet1_precision = (precision_eval(unet_1, validation_images, validation_masks))
am_unet_precision = (precision_eval(am_unet, validation_images, validation_masks))
am_unet1_precision = (precision_eval(am_unet_1, validation_images, validation_masks))
unet_large_precision = (precision_eval(unet_large, validation_images2, validation_masks2))
am_unet_large_precision = (precision_eval(am_unet_large, validation_images2, validation_masks2))

print('Calculating recall statistics...')
unet_recall = (recall_eval(unet, validation_images, validation_masks))
unet1_recall = (recall_eval(unet_1, validation_images, validation_masks))
am_unet_recall = (recall_eval(am_unet, validation_images, validation_masks))
am_unet1_recall = (recall_eval(am_unet_1, validation_images, validation_masks))
unet_large_recall = (recall_eval(unet_large, validation_images2, validation_masks2))
am_unet_large_recall = (recall_eval(am_unet_large, validation_images2, validation_masks2))

# F1-scores of each model
print('Calculating F1-score statistics...')
unet_f1_score = (f1_score_eval_basic(unet_precision, unet_recall))
unet1_f1_score = (f1_score_eval_basic(unet1_precision, unet1_recall))
am_unet_f1_score = (f1_score_eval_basic(am_unet_precision, am_unet_recall))
am_unet1_f1_score = (f1_score_eval_basic(am_unet1_precision, am_unet1_recall))
unet_large_f1_score = (f1_score_eval_basic(unet_large_precision, unet_large_recall))
am_unet_large_f1_score = (f1_score_eval_basic(am_unet_large_precision, am_unet_large_recall))

# Print score eval results for each model
print('Printing scores...')
print(np.mean(unet_score))
print(np.mean(unet1_score))
print(np.mean(am_unet_score))
print(np.mean(am_unet1_score))
print(np.mean(unet_large_score))
print(np.mean(am_unet_large_score))

# Print precision eval results for each model
print('Printing precision statistics...')
print(np.mean(unet_precision))
print(np.mean(unet1_precision))
print(np.mean(am_unet_precision))
print(np.mean(am_unet1_precision))
print(np.mean(unet_large_precision))
print(np.mean(am_unet_large_precision))

# Print recall eval results for each model
print('Printing recall statistics...')
print(np.mean(unet_recall))
print(np.mean(unet1_recall))
print(np.mean(am_unet_recall))
print(np.mean(am_unet1_recall))
print(np.mean(unet_large_recall))
print(np.mean(am_unet_large_recall))

# Print f1-score eval results for each model
print('Printing F1-score statistics...')
print(np.mean(unet_f1_score))
print(np.mean(unet1_f1_score))
print(np.mean(am_unet_f1_score))
print(np.mean(am_unet1_f1_score))
print(np.mean(unet_large_f1_score))
print(np.mean(am_unet_large_f1_score))

