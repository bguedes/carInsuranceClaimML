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

def save_bottleneck_features():
    datagen = ImageDataGenerator(rescale=1./255)

    model = VGG16(
        include_top=False,
        weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    bottleneck_features_train = model.predict(
        generator,
        nb_train_samples // batch_size)

    np.save(location+'/bottleneck_features_train.npy', bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    bottleneck_features_validation = model.predict(
        generator,
        nb_validation_samples // batch_size)

    np.save(location+'/bottleneck_features_validation.npy', bottleneck_features_validation)

def train_top_model():
    train_data = np.load(location+'/bottleneck_features_train.npy')
    train_labels = np.array([0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))

    validation_data = np.load(location+'/bottleneck_features_validation.npy')
    validation_labels = np.array([0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(224,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
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
        batch_size=batch_size,
        validation_data=(validation_data,validation_labels),
        callbacks=[checkpoint])

    with open(location+'/top_history.txt', 'w') as f:
        json.dump(fit.history, f)

    return model, fit.history

def finetune_binary_model():
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224,224,3))

    print("Model loaded.")

    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(224, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    top_model.load_weights(top_model_weights_path)

    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

    for layer in model.layers[:25]:
        layer.trainable = False

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.9),
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
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

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

    with open(location+'/ft_history.txt', 'w') as f:
        json.dump(fit.history, f)

    return model, fit.history

train_datagen = ImageDataGenerator(rescale=1/255.)
validation_datagen = ImageDataGenerator(rescale=1/255.)

location = '../data/carDamage'
top_model_weights_path = location+'../models/topDamageModelWeights.h5'
fine_tuned_model_path = location+'../models/carDamagePredictionModel.h5'

train_data_dir = location+'/training'
validation_data_dir = location+'/validation'
train_samples = [len(os.listdir(train_data_dir+'/'+i)) for i in sorted(os.listdir(train_data_dir))]
nb_train_samples = 1840
validation_samples = [len(os.listdir(validation_data_dir+'/'+i)) for i in sorted(os.listdir(validation_data_dir))]
nb_validation_samples = 460

img_width, img_height = 224,224
epochs = 20
batch_size = 16

save_bottleneck_features()

d2_model1, d2_history1 = train_top_model()

ft_model, ft_history = finetune_binary_model()
