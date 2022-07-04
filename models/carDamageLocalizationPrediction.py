import tensorflow as tf
import base64
import numpy as np

from keras.applications.vgg16 import VGG16
from tensorflow import keras

def load_and_prep_image(imageBase64Encoded, img_shape=224):
  img = base64.decodebytes(imageBase64Encoded.encode())
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.resize(img, size = [img_shape, img_shape])
  img = img/255.
  return img

def localizationPrediction(imageBase64Encoded, model):
    img = load_and_prep_image(imageBase64Encoded)
    pred = model.predict(tf.expand_dims(img, axis=0))
    pred_labels = np.argmax(pred, axis=1)
    locationArray = {0:'Front', 1:'Rear', 2:'Side'}
    for key in locationArray.keys():
        if pred_labels[0] == key:
            return {"localization" : locationArray[key]}

def detectDamageLocalization(args):
    imageBase64Encoded = args["imageBase64"]
    model = keras.models.load_model('carDamageLocalizationPredictionModel.h5')
    return localizationPrediction(imageBase64Encoded, model)
