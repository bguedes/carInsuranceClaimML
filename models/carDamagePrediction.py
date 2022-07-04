import tensorflow as tf
import base64
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os

from keras.layers import Flatten, Dense
from tensorflow.keras.applications import VGG16
from tensorflow import keras

def load_and_prep_image(base64ImageEncoded, img_shape=224):
  print(base64ImageEncoded)
  img = base64.decodebytes(base64ImageEncoded.encode())
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.resize(img, size = [img_shape, img_shape])
  img = img/255.
  return img

def pred_and_plot(model, base64ImageEncoded):
  img = load_and_prep_image(base64ImageEncoded)
  pred = model.predict(tf.expand_dims(img, axis=0))
  pred=pred.round()
  if pred==0:
    return {"car_damaged": "true"}
  else:
      return {"car_damaged": "false"}

def detectIfCarIsImage():
    imageBase64Encoded = args["imageBase64"]
    model = keras.models.load_model('carDamagePrediction.h5')
    return pred_and_plot(model, imageBase64Encoded)
