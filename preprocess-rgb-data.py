import numpy as np
import os
import PIL
from PIL import Image

#
# Pre-Process 'Amazon Forest Dataset' dataset
#

base_dir = r"./Amazon Forest Dataset/"
## Training images
training_images_list = os.listdir(r"{}Training/images/".format(base_dir))
training_masks_list = []
training_images = []
for n in training_images_list:
  im = PIL.Image.open(r"{}Training/images/{}".format(base_dir,n))
  training_images.append(im)
  training_masks_list.append(n[:-5]+'.png')

## Training masks
training_masks = []
for n in training_masks_list:
  im = PIL.Image.open(r"{}Training/masks/{}".format(base_dir,n))
  training_masks.append(im)

## Test images
test_images_list = os.listdir(r"{}Test/".format(base_dir))
test_images = []
for n in test_images_list:
  im = PIL.Image.open(r"{}Test/{}".format(base_dir,n))
  test_images.append(im)

## Validation images
validation_images_list = os.listdir(r"{}Validation/images/".format(base_dir))
validation_masks_list = []
validation_images = []
for n in validation_images_list:
  im = PIL.Image.open(r"{}Validation/images/{}".format(base_dir,n))
  validation_images.append(im)
  validation_masks_list.append(n[:-5]+'.png')

## Validation masks
validation_masks = []
for n in validation_masks_list:
  im = PIL.Image.open(r"{}Validation/masks/{}".format(base_dir,n))
  validation_masks.append(im)

# Pre-process data
for i in range(len(training_images)):
  training_images[i] = np.array(training_images[i])/255
  training_images[i] = training_images[i].reshape(1,512,512,3)
  training_images[i] = training_images[i].astype('float32')

for i in range(len(training_masks)):
  training_masks[i] = (np.array(training_masks[i])-1)
  training_masks[i] = training_masks[i][:512,:512]
  training_masks[i] = training_masks[i].reshape(1,512,512,1)
  training_masks[i] = training_masks[i].astype('int')

for i in range(len(validation_images)):
  validation_images[i] = np.array(validation_images[i])/255
  validation_images[i] = validation_images[i].reshape(1,512,512,3)
  validation_images[i] = validation_images[i].astype('float32')

for i in range(len(validation_masks)):
  validation_masks[i] = np.array(validation_masks[i])-1
  validation_masks[i] = validation_masks[i][:512,:512]
  validation_masks[i] = validation_masks[i].reshape(1,512,512,1)
  validation_masks[i] = validation_masks[i].astype('int')

for i in range(len(test_images)):
  test_images[i] = np.array(test_images[i])/255
  test_images[i] = test_images[i].reshape(1,512,512,3)
  test_images[i] = test_images[i].astype('float32')


#
# Output as .npy files
#
os.mkdir('amazon-processed-regular')
os.chdir('amazon-processed-regular')

os.mkdir('training')
os.mkdir('test')
os.mkdir('validation')

os.chdir('training')
os.mkdir('images')
os.mkdir('masks')
os.chdir('images')
for i in range(len(training_images)):
    np.save(training_images_list[i], training_images[i])
os.chdir('../masks')
for i in range(len(training_masks)):
    np.save(training_masks_list[i], training_masks[i])
os.chdir('..')


os.chdir('../test')
os.mkdir('images')
os.chdir('images')
for i in range(len(test_images)):
    np.save(test_images_list[i], test_images[i])
os.chdir('..')

os.chdir('../validation')
os.mkdir('images')
os.mkdir('masks')
os.chdir('images')
for i in range(len(validation_images)):
    np.save(validation_images_list[i], validation_images[i])
os.chdir('../masks')
for i in range(len(validation_masks)):
    np.save(validation_masks_list[i], validation_masks[i])
os.chdir('..')



