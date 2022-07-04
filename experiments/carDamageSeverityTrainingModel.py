import tensorflow as tf
import numpy as np
import json

import random
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from tensorflow import keras
from keras.utils.np_utils import to_categorical

def save_bottleneck_features():
    datagen = ImageDataGenerator(rescale=1./255)

    model = VGG16(
        include_top=False,
        weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode=None,
        shuffle=False)

    bottleneck_features_train = model.predict(
        generator,
        nb_train_samples // 1)

    np.save(location+'/bottleneck_features_train.npy', bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode=None,
        shuffle=False)

    bottleneck_features_validation = model.predict(
        generator,
        nb_validation_samples // 1)

    np.save(location+'/bottleneck_features_validation.npy', bottleneck_features_validation)

def train_categorical_model():

    train_data = np.load(location+'/bottleneck_features_train.npy')
    train_labels = np.array([0]*(278) + [1]*(315) + [2]*(386))
    train_labels = to_categorical(train_labels)

    validation_data = np.load(location+'/bottleneck_features_validation.npy')
    validation_labels = np.array([0]*(48) + [1]*(55) + [2]*(68))
    validation_labels = to_categorical(validation_labels)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(224,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3,activation='softmax'))

    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    checkpoint = ModelCheckpoint(
        top_model_weights_path,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='auto')

    fit = model.fit(
        train_data,
        train_labels,
        epochs=epochs,
        batch_size=1,
        validation_data=(validation_data,validation_labels),
        callbacks=[checkpoint])

    with open(location+'/top_damage_location_history.txt', 'w') as f:
        json.dump(fit.history, f)

    return model, fit.history

def finetune_categorical_model():
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224,224,3))

    print("Model loaded.")

    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(224, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(3, activation='softmax'))

    top_model.load_weights(top_model_weights_path)

    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

    for layer in model.layers[:25]:
        layer.trainable = False

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.SGD(learning_rate=0.00001, momentum=0.9),
        metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rescale = 1./255,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    checkpoint = ModelCheckpoint(
        fine_tuned_model_path,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto')

    fit = model.fit(
        train_generator,
        steps_per_epoch=nb_train_samples//batch_size,
        epochs=epochs, validation_data=validation_generator,
        validation_steps=nb_validation_samples//batch_size,
        verbose=1,
        callbacks=[checkpoint])

    with open(location+'/ft_damage_location_history.txt', 'w') as f:
        json.dump(fit.history, f)

    return model, fit.history

train_datagen = ImageDataGenerator(rescale=1/255.)
validation_datagen = ImageDataGenerator(rescale=1/255.)

location = '../data/damageSeverity'
top_model_weights_path = '../models/top_damage_location_categorical_weights.h5'
fine_tuned_model_path = '../models/carDamageSeverityPredictionModel.h5'

train_data_dir = location+'/training'
validation_data_dir = location+'/validation'
train_samples = [len(os.listdir(train_data_dir+'/'+i)) for i in sorted(os.listdir(train_data_dir))]
nb_train_samples = 979
validation_samples = [len(os.listdir(validation_data_dir+'/'+i)) for i in sorted(os.listdir(validation_data_dir))]
nb_validation_samples = 171

img_width, img_height = 224,224
epochs = 50
batch_size = 16

save_bottleneck_features()

d2_model1, d2_history1 = train_categorical_model()

ft_model, ft_history = finetune_categorical_model()
