from ast import arg
from importlib.resources import path
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
#from colorthief import ColorThief
import csv
import cv2
from yaml import parse
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import shutil
import argparse
from keras.models import load_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input, BatchNormalization, Lambda, AveragePooling2D, ZeroPadding2D, Reshape, Activation, Add, GlobalAveragePooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm


#pet_classes = os.listdir('/home/henishv5/WI_Testing/pet_images_new/')
"""
os.mkdir('/home/henishv5/WI_Testing/test_pet/')

os.mkdir('/home/henishv5/WI_Testing/test_pet/test/')
os.mkdir('/home/henishv5/WI_Testing/test_pet/train/')
os.mkdir('/home/henishv5/WI_Testing/test_pet/val/')
os.mkdir('/home/henishv5/WI_Testing/test_pet/test/crushed/')
os.mkdir('/home/henishv5/WI_Testing/test_pet/train/crushed/')
os.mkdir('/home/henishv5/WI_Testing/test_pet/val/crushed/')
os.mkdir('/home/henishv5/WI_Testing/test_pet/test/uncrushed/')
os.mkdir('/home/henishv5/WI_Testing/test_pet/train/uncrushed/')
os.mkdir('/home/henishv5/WI_Testing/test_pet/val/uncrushed/')
"""


def is_image(img_):
    if img_.split('.')[-1] in ['png', 'jpg', 'jpeg', 'raw']:
        return True
    else:
        return False


def copy_to_folder(list_of_files, dest_path):
    for f in list_of_files:
        try:
            shutil.copy(f, dest_path)
        except Exception as e:
            print(e)

def move_to_folder(file_, dest_path):
    try:
        shutil.move(file_, dest_path)
    except Exception as e:
        print(e)

def train_pet_classifier():
    for p_c in pet_classes:
        images = []
        for f in os.listdir('/home/henishv5/WI_Testing/pet_images_new/' + p_c+'/'):
            if is_image(f):
                images.append(
                    '/home/henishv5/WI_Testing/pet_images_new/' + p_c+'/' + f)
        train_images, val_images = train_test_split(
            images, test_size=0.2, random_state=1)
        val_images, test_images = train_test_split(
            val_images, test_size=0.5, random_state=1)
        copy_to_folder(
            train_images, '/home/henishv5/WI_Testing/test_pet/test/' + p_c + '/')
        copy_to_folder(
            val_images, '/home/henishv5/WI_Testing/test_pet/val/' + p_c + '/')
        copy_to_folder(
            test_images, '/home/henishv5/WI_Testing/test_pet/train/' + p_c + '/')

    base_dir = '/home/henishv5/WI_Testing/test_pet/'

    # ImageDataGenerator for training
    train_datagen = ImageDataGenerator(rescale=1./255
                                       # shear_range=0.2,
                                       # zoom_range=0.3,
                                       # horizontal_flip=True,
                                       # brightness_range=[0.5,1.5],
                                       )

    train_generator = train_datagen.flow_from_directory(
        os.path.join(base_dir, 'train'),
        #directory = None,
        #x_col="image path",
        # y_col="car_label",
        # has_ext=True,
        # subset="training",
        batch_size=32,
        # seed=42,
        # shuffle=True
        class_mode="categorical",
        target_size=(56, 56),
        color_mode='rgb'
    )

    # ImageDataGenerator for val
    val_datagen = ImageDataGenerator(rescale=1./255
                                     # shear_range=0.2,
                                     # zoom_range=0.3,
                                     # horizontal_flip=True,
                                     # brightness_range=[0.5,1.5],
                                     )

    valid_generator = val_datagen.flow_from_directory(
        os.path.join(base_dir, 'val'),
        #directory = None,
        #x_col="image path",
        # y_col="car_label",
        # has_ext=True,
        # subset="validation",
        batch_size=32,
        # seed=42,
        # shuffle=True,
        class_mode="categorical",
        target_size=(56, 56),
        color_mode='rgb'
    )

    # ImageDataGenerator for testing
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        os.path.join(base_dir, 'test'),

        #directory = None,
        #x_col="image path",
        # y_col=None, # None for testing
        #     has_ext=True,
        batch_size=32,
        # seed=42,
        # shuffle=False, # yield images in order
        class_mode=None,  # return only the images
        target_size=(56, 56),
        color_mode='rgb'
    )

    # Defining epoch step sizes based on number of images divided by batch size
    step_train = train_generator.n//train_generator.batch_size
    step_valid = valid_generator.n//valid_generator.batch_size
    step_test = test_generator.n//test_generator.batch_size

    def pet_class():
        def res_net_block(input_data, filters, conv_size):
            x = Conv2D(filters, conv_size, activation='relu',
                       padding='same')(input_data)
            x = BatchNormalization()(x)
            x = Conv2D(filters, conv_size, activation=None, padding='same')(x)
            x = BatchNormalization()(x)
            x = Add()([x, input_data])
            x = Activation('relu')(x)
            return x

        inputs = keras.Input(shape=(56, 56, 3))
        x = Conv2D(64, 3, activation='relu')(inputs)
        x = Conv2D(128, 3, activation='relu')(x)
        x = MaxPooling2D(3)(x)
        num_res_net_blocks = 2
        input_filter = 128
        for i in range(num_res_net_blocks):
            x = res_net_block(x, input_filter, 3)
        x = Conv2D(128, 3, activation='relu')(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        #x = layers.Dropout(0.5)(x)
        outputs = Dense(2, activation='softmax')(x)
        res_net_model = keras.Model(inputs, outputs)

        res_net_model.compile(optimizer=keras.optimizers.Adam(),
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])
        return res_net_model

    # initialise model
    model = pet_class()

    # Model Summary
    model.summary()
    # os.mkdir('/home/henishv5/WI_Testing/pet_models_v3/')
    model.load_weights(
        '/home/henishv5/WI_Testing/pet_models_v2/pet_weights.hdf5')

    nb_epoch = 100

    filepath = '/home/henishv5/WI_Testing/pet_models_v2/pet_weights.hdf5'
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]

    history = model.fit(
        train_generator,
        steps_per_epoch=step_train,
        epochs=nb_epoch,
        validation_data=valid_generator,
        validation_steps=step_valid,
        callbacks=callbacks_list,
        verbose=1)

    model.save('/home/henishv5/WI_Testing/pet_models_v3/ep_100')

    model_builder = keras.applications.xception.Xception
    img_size = (56, 56)
    preprocess_input = keras.applications.xception.preprocess_input
    decode_predictions = keras.applications.xception.decode_predictions

    # Evaluate model
    model = keras.models.load_model(
        '/home/henishv5/WI_Testing/pet_models_v2/pet_weights.hdf5')
    model.evaluate(
        valid_generator,
        steps=step_valid,
        verbose=1)


def predict_class(dest_path, image_, cs):
    
    if cs == False:
        img = tf.keras.utils.load_img(image_, target_size=(56, 56))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
            
        model = load_model(
            '/home/henishv5/WI_Testing/pet_models_v3/ep_100')
        predictions = model.predict(img_array, verbose = 0)
        if predictions[0][0] == 1:
            x = 'crushed'
        else:
            x = 'uncrushed'
        #cv2.imread(image_)
        cv2.imshow(x, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if cs ==True:
        if 'crushed' not in os.listdir(dest_path):
            os.mkdir(dest_path+'/crushed')
        if 'uncrushed' not in os.listdir(dest_path):
            os.mkdir(dest_path+'/uncrushed')
        
        img = tf.keras.utils.load_img(image_, target_size=(56, 56))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

            
        model = load_model(
            '/home/henishv5/WI_Testing/pet_models_v3/ep_100')
        predictions = model.predict(img_array, verbose = 0)
        if predictions[0][0] == 1:
            x = 'crushed'
        else:
            x = 'uncrushed'
        
        copy_to_folder(image_, dest_path + x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', help='Provide image path here to classify the image', required = True)
    parser.add_argument('--dest_path', help='Provide path to store images into crushed uncrushed folders', required = True)
    parser.add_argument('--crop_save', help='True if crops are already saved.', required= True, default= False, type=bool)
    args = parser.parse_args()
    src_path = str(args.src_path)
    dest_path = args.dest_path
    if os.path.isdir(src_path):
        for f in os.listdir(src_path):
            if is_image(f):
                predict_class(dest_path, src_path+f, args.crop_save)
            else:
                pass
    elif is_image(src_path):
        predict_class(dest_path, src_path, args.crop_save)
        # predict_class(src_path)
    # predict_class('/home/henishv5/WI_Testing/test/data_final/crops/pet_bottles/frame_3202_0_1023.png')
    # predict_class('/home/henishv5/WI_Testing/pet_images_new/uncrushed/image_477.png')
    #train_pet_classifier()