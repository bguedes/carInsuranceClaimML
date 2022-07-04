import tensorflow as tf
import base64
import numpy as np
from keras.applications.vgg16 import VGG16
from tensorflow import keras
from tensorflow.keras.utils import img_to_array

def load_and_prep_image(imageBase64Encoded, img_shape=224):
  img = base64.decodebytes(imageBase64Encoded.encode())
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.resize(img, size = [img_shape, img_shape])
  img = img/255.
  return img

def damagePrediction(imageBase64Encoded, model):
    img = load_and_prep_image(imageBase64Encoded)
    pred = model.predict(tf.expand_dims(img, axis=0))
    pred_labels = np.argmax(pred, axis=1)
    severityArray = {0:'Minor', 1:'Moderate', 2:'Severe'}
    for key in severityArray.keys():
        if pred_labels[0] == key:
            print("Validating severity of damage...Result:",severityArray[key])
    print("Severity assessment complete.")

def detectCarImage(args):
    imageBase64Encoded = args["imageBase64"]
    model = keras.models.load_model('carDamageSeverityPredictionModel.h5')
    return damagePrediction(imageBase64Encoded, model)
