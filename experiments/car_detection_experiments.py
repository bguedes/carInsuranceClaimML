# Import modules:
import csv
import io
import json
import os
import pickle as pk
import sys
import urllib.request
from collections import Counter, defaultdict
from io import BytesIO

import cdsw
import efficientnet.keras as efn
import keras
import numpy as np
import requests
from joblib import dump
from keras.applications import imagenet_utils
from keras.applications.vgg16 import VGG16
from keras.utils.data_utils import get_file
from PIL import Image
from tensorflow.keras.utils import img_to_array, load_img

CLASS_INDEX = None

def load_model():
    global model
    model = VGG16(input_shape=(224, 224,3),
                  weights="imagenet")

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image

def get_predictions(preds, top=5):
    global CLASS_INDEX
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects a batch of predictions (i.e. a 2D array of shape (samples, 1000)). Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        fpath = get_file('imagenet_class_index.json',
            'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json',
            cache_subdir='models')
        CLASS_INDEX = json.load(open(fpath))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results

def get_car_categories():
    d = defaultdict(float)
    img_list = os.listdir('data/carDetection/training/01-whole/')
    for i, img_path in enumerate(img_list):
        sys.stdout.write('img_path : ' + img_path)
        image = load_img('data/carDetection/training/01-whole/' + img_path, target_size=(224, 224))
        img = prepare_image(image, target=(224, 224))
        out = model.predict(img)
        preds = get_predictions(out,top=5)
        for pred in preds[0]:
            d[pred[0:2]]+=pred[2]
        if(i%50==0):
            print(i,'/',len(img_list),'complete')

    print('d -> ', d)
    return Counter(d)

load_model()
categ_count = get_car_categories()

dump(categ_count, 'car_model_cat_list.pk')
cdsw.track_file('car_model_cat_list.pk')
