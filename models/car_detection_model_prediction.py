import base64
import json
import pickle as pk
import io
import keras
import numpy as np
import requests
from joblib import load
from keras.applications import ResNet50, imagenet_utils
from keras.applications.vgg16 import VGG16
from keras.utils.data_utils import get_file
from PIL import Image
from tensorflow.keras.utils import img_to_array, load_img

CLASS_INDEX = None

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
        #fpath = get_file('imagenet_class_index.json',
        #    'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json',
        #    cache_subdir='models')
        #CLASS_INDEX = json.load(open(fpath))
        CLASS_INDEX = json.load(open('/home/cdsw/models/imagenet_class_index.json'))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results

def predict(imageBase64Encoded, categ_list):
    image = Image.open(io.BytesIO(base64.decodebytes(imageBase64Encoded.encode())))   
    img = prepare_image(image, target=(224, 224))
    model = VGG16(input_shape=(224, 224,3), weights="imagenet")
    out = model.predict(img)
    preds = get_predictions(out, top=5)
    for pred in preds[0]:
        if pred[0:2] in categ_list:
            return {"car_detected": "true"}
    return {"car_detected": "false"}

def detectCarImage(args):
    imageBase64Encoded = args["imageBase64"] 
    with open('/home/cdsw/models/car_model_cat_list.pk', 'rb') as f:
        categ_count = pk.load(f)
    categ_list = [k for k, v in categ_count.most_common()[:25]]
    return predict(imageBase64Encoded, categ_list)
