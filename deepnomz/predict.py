import tensorflow as tf
tf.python.control_flow_ops = tf
import os
import h5py
import numpy as np
from keras.preprocessing import image
from keras.models import *
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import VGG16

# path to the model weights file.
path_to_test_img = '/Users/manas/gdrive/work/project/deepnomz/test/'
path_to_saved_models = '/Users/manas/gdrive/work/project/deepnomz/saved_models/'
vgg16model_path = path_to_saved_models + 'vgg16model.h5'
topmodel_path = path_to_saved_models + 'topmodel.h5'

# dimensions of our images.
img_width, img_height = 150, 150
img_path = path_to_test_img + 'test1.jpg'


img = image.load_img(img_path, target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

vgg16model = load_weights(vgg16model_path)
topmodel = load_weights(topmodel_path)
full_model = vgg16model.add(topmodel)

class_preds = full_model.predict_classes(x)
prob_preds = full_model.predict_proba(x)
print('Predicted classes:', decode_predictions(class_preds))
print('Predicted probabilities:', decode_predictions(prob_preds))
