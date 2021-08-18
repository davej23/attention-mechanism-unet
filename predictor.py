#
# Produces an output deforestation mask
#

# Import packages
import tensorflow as tf
import keras
from keras import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras import backend as K
import sys
#from sklearn.metrics import *

# Import image and model
image_dir = sys.argv[2]

image = Image.open(image_dir)
image_array = np.array(image)/255

model1 = load_model('unet-attention-3d.hdf5')
model2 = load_model('unet-attention-4d.hdf5')
model3 = load_model('unet-attention-4d-atlantic.hdf5')
models = [model1, model2, model3]

# Activate the specified model, according to sys.argv[1]
model = models[int(sys.argv[1])]

# Process image to be in 512x512 chunks
def resize_image(im, input_array):
    input_shape = input_array.shape

    x = (input_shape[0] % 512)
    y = (input_shape[1] % 512)

    output1 = np.pad(input_array[:,:,0], [[0, 512-x], [0, 512-y]], mode='constant', constant_values=0)
    output2 = np.pad(input_array[:,:,1], [[0, 512-x], [0, 512-y]], mode='constant', constant_values=0)
    output3 = np.pad(input_array[:,:,2], [[0, 512-x], [0, 512-y]], mode='constant', constant_values=0)
    output = np.zeros((output1.shape[0], output1.shape[1], 3))
    output[:,:,0] = output1
    output[:,:,1] = output2
    output[:,:,2] = output3

    return output

padded_image = resize_image(image, image_array)

split_images = []

for i in range(padded_image.shape[0]):
    for j in range(padded_image.shape[1]):
        split_images.append(padded_image[i:i+512, j:j+512, :])


# Predict mask
predictions = []
for n in split_images:
    predictions.append(model.predict(n.reshape(1,512,512,3)))

out = np.zeros((padded_image.shape))

for i in range(padded_image.shape[0]):
    for j in range(padded_image.shape[1]):
        if len(predictions) > 0:
            out[i:i+512, j:j+512, :] = predictions[0]
            del predictions[0]


# Save to file
plt.imsave('deforestation-mask.png', out)