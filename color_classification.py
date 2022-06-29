#Importing Libraries

from copy import copy
from matplotlib import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import shutil
import argparse
import cv2
from keras.models import load_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input, BatchNormalization, Lambda, AveragePooling2D, ZeroPadding2D, Reshape, Activation, Concatenate
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

from IPython.display import Image, display
import matplotlib.cm as cm

def is_image(file_format):
  if file_format.split('.')[-1] in ['png', 'jpeg', 'jpg', 'raw']:
    return True
  else:
    return False

#<--- ONLY FOR ME --->#
"""
  This lines of code to be used only when we want to train data
  It will be used to create dirctory for Train Test Val and sub-catagories of colors under them
"""


ttv_cls = ['train/', 'test/', 'val/']
color_cls = ['black', 'blue', 'green', 'red',  'transparent', 'white', 'yellow']
#UNCOMMENT IF REQUIRED TO RUN
"""
for cls in ttv_cls:
  os.mkdir('/home/henishv5/WI_Testing/color_dataset/' + cls)
  for c_cls in color_cls:
    os.mkdir('/home/henishv5/WI_Testing/color_dataset/' + cls + c_cls)
"""
#Function to copy file to the destined folder

def copy_to_folder(list_of_files, dest_path):
    for f in list_of_files:
        try:
            shutil.copy(f, dest_path)
        except Exception as e:
            print(e)

def get_py_path():
    return str(os.getcwd()) + '/'

#Function to copy file to the destined folder

def move_to_folder(list_of_files, dest_path):
  for f in list_of_files:
    try:
      shutil.move(f, dest_path)
    except Exception as e:
      print(e)


def train_color_model():
    #To divide dataset into training set, validation set, test set
    """
    for c_cls in color_cls:
      data_images = []
      for i in os.listdir('/home/henishv5/WI_Testing/color_dataset/' + c_cls):
        print(i)
        if is_image(i):
          data_images.append('/home/henishv5/WI_Testing/color_dataset/' + c_cls + i)
      data_images.sort()
      print(data_images)
      train_images, val_images = train_test_split(data_images, test_size = 0.2, random_state = 1)
      val_images, test_images = train_test_split(val_images, test_size = 0.5, random_state = 1)
      print(len(val_images))
      move_to_folder(train_images, '/home/henishv5/WI_Testing/color_dataset/train/' + c_cls)
      move_to_folder(test_images, '/home/henishv5/WI_Testing/color_dataset/test/' + c_cls)
      move_to_folder(val_images, '/home/henishv5/WI_Testing/color_dataset/val/' + c_cls)
      """
    dir_ = '/home/henishv5/WI_Testing/color_dataset/'

    # ImageDataGenerator for training
    train_datagen = ImageDataGenerator(rescale=1./255     
                                #shear_range=0.2,
                                #zoom_range=0.3,
                                #horizontal_flip=True,
                                # brightness_range=[0.5,1.5],
                                )

    train_generator = train_datagen.flow_from_directory(
        os.path.join(dir_ ,'train'),
        #directory = None,
        #x_col="image path",
        #y_col="car_label",
        #has_ext=True,                                     
        #subset="training",
        batch_size=64,
        #seed=42,
        #shuffle=True
        class_mode="categorical",
        target_size=(56,56), 
        color_mode='rgb'
    )

    # ImageDataGenerator for val
    val_datagen = ImageDataGenerator(rescale=1./255     
                                #shear_range=0.2,
                                #zoom_range=0.3,
                                #horizontal_flip=True,
                                #brightness_range=[0.5,1.5],
                                )

    valid_generator = val_datagen.flow_from_directory(
        os.path.join(dir_,'val'),
        #directory = None, 
        #x_col="image path",
        #y_col="car_label",
        #has_ext=True,
        #subset="validation",
        batch_size=64,
        #seed=42,
        #shuffle=True,
        class_mode="categorical",
        target_size=(56, 56), 
        color_mode='rgb'
    )

    # ImageDataGenerator for val
    test_datagen = ImageDataGenerator(rescale=1./255     
                                #shear_range=0.2,
                                #zoom_range=0.3,
                                #horizontal_flip=True,
                                #brightness_range=[0.5,1.5],
                                )

    test_generator = test_datagen.flow_from_directory(
        os.path.join(dir_,'test'),
        #directory = None, 
        #x_col="image path",
        #y_col="car_label",
        #has_ext=True,
        #subset="validation",
        batch_size=64,
        #seed=42,
        #shuffle=True,
        class_mode="categorical",
        target_size=(56, 56), 
        color_mode='rgb'
    )

    # Defining epoch step sizes based on number of images divided by batch size
    step_train = train_generator.n//train_generator.batch_size
    step_valid = valid_generator.n//valid_generator.batch_size
    step_test = test_generator.n//test_generator.batch_size

    def color_net(num_classes):
        # placeholder for input image
        input_image = Input(shape=(56,56,3))
        # ============================================= TOP BRANCH ===================================================
        # first top convolution layer
        top_conv1 = Conv2D(filters=48,kernel_size=(11,11),strides=(4,4),
                                input_shape=(56,56,3),activation='relu')(input_image)
        top_conv1 = BatchNormalization()(top_conv1)
        top_conv1 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(top_conv1)

        # second top convolution layer
        # split feature map by half
        top_top_conv2 = Lambda(lambda x : x[:,:,:,:24])(top_conv1)
        top_bot_conv2 = Lambda(lambda x : x[:,:,:,24:])(top_conv1)

        top_top_conv2 = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_top_conv2)
        top_top_conv2 = BatchNormalization()(top_top_conv2)
        top_top_conv2 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(top_top_conv2)

        top_bot_conv2 = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_bot_conv2)
        top_bot_conv2 = BatchNormalization()(top_bot_conv2)
        top_bot_conv2 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(top_bot_conv2)

        # third top convolution layer
        # concat 2 feature map
        top_conv3 = Concatenate()([top_top_conv2,top_bot_conv2])
        top_conv3 = Conv2D(filters=192,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_conv3)

        # fourth top convolution layer
        # split feature map by half
        top_top_conv4 = Lambda(lambda x : x[:,:,:,:96])(top_conv3)
        top_bot_conv4 = Lambda(lambda x : x[:,:,:,96:])(top_conv3)

        top_top_conv4 = Conv2D(filters=96,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_top_conv4)
        top_bot_conv4 = Conv2D(filters=96,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_bot_conv4)

        # fifth top convolution layer
        top_top_conv5 = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_top_conv4)
        top_top_conv5 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(top_top_conv5) 

        top_bot_conv5 = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_bot_conv4)
        top_bot_conv5 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(top_bot_conv5)

        # ============================================= TOP BOTTOM ===================================================
        # first bottom convolution layer
        bottom_conv1 = Conv2D(filters=48,kernel_size=(11,11),strides=(4,4),
                                input_shape=(227,227,3),activation='relu')(input_image)
        bottom_conv1 = BatchNormalization()(bottom_conv1)
        bottom_conv1 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(bottom_conv1)

        # second bottom convolution layer
        # split feature map by half
        bottom_top_conv2 = Lambda(lambda x : x[:,:,:,:24])(bottom_conv1)
        bottom_bot_conv2 = Lambda(lambda x : x[:,:,:,24:])(bottom_conv1)

        bottom_top_conv2 = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_top_conv2)
        bottom_top_conv2 = BatchNormalization()(bottom_top_conv2)
        bottom_top_conv2 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(bottom_top_conv2)

        bottom_bot_conv2 = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_bot_conv2)
        bottom_bot_conv2 = BatchNormalization()(bottom_bot_conv2)
        bottom_bot_conv2 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(bottom_bot_conv2)

        # third bottom convolution layer
        # concat 2 feature map
        bottom_conv3 = Concatenate()([bottom_top_conv2,bottom_bot_conv2])
        bottom_conv3 = Conv2D(filters=192,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_conv3)

        # fourth bottom convolution layer
        # split feature map by half
        bottom_top_conv4 = Lambda(lambda x : x[:,:,:,:96])(bottom_conv3)
        bottom_bot_conv4 = Lambda(lambda x : x[:,:,:,96:])(bottom_conv3)

        bottom_top_conv4 = Conv2D(filters=96,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_top_conv4)
        bottom_bot_conv4 = Conv2D(filters=96,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_bot_conv4)

        # fifth bottom convolution layer
        bottom_top_conv5 = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_top_conv4)
        bottom_top_conv5 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(bottom_top_conv5) 

        bottom_bot_conv5 = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_bot_conv4)
        bottom_bot_conv5 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(bottom_bot_conv5)

        # ======================================== CONCATENATE TOP AND BOTTOM BRANCH =================================
        conv_output = Concatenate()([top_top_conv5,top_bot_conv5,bottom_top_conv5,bottom_bot_conv5])

        # Flatten
        flatten = Flatten()(conv_output)

        # Fully-connected layer
        FC_1 = Dense(units=1024, activation='relu')(flatten)
        FC_1 = Dropout(0.6)(FC_1)
        #FC_2 = Dense(units=4096, activation='relu')(FC_1)
        #FC_2 = Dropout(0.6)(FC_2)
        output = Dense(units=num_classes, activation='softmax')(FC_1)
        
        model = Model(inputs=input_image,outputs=output)
        sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        # sgd = SGD(lr=0.01, momentum=0.9, decay=0.0005, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model


    # initialise model
    num_classes = 7
    model = color_net(num_classes)

    nb_epoch = 50

    #os.mkdir('/home/henishv5/WI_Testing/color_models_v1/')
    #os.mkdir('/home/henishv5/WI_Testing/color_models_v1/ep_50')

    filepath = '/home/henishv5/WI_Testing/color_model_v1/color_weights.hdf5'
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

    

    model.save('/home/henishv5/WI_Testing/color_models_v1/ep_50')

    # Evaluate model
    model.evaluate(
        valid_generator, 
        steps=step_valid, 
        verbose=1)

    # Model Summary
    model.summary()

    # Check out our train loss and test loss over epochs.
    train_loss = history.history['loss']
    test_loss = history.history['val_loss']

    # Set figure size.
    plt.figure(figsize=(12, 8))

    # Generate line plot of training, testing loss over epochs.
    plt.plot(train_loss, label='Training Loss', color='#185fad')
    plt.plot(test_loss, label='Testing Loss', color='orange')

    # Set title
    plt.title('Training and Testing Loss by Epoch(50)', fontsize = 25)
    plt.xlabel('Epoch', fontsize = 18)
    plt.ylabel('Categorical Crossentropy', fontsize = 18)
    plt.xticks(range(15))

    plt.legend(fontsize = 18);

    # Check out our train loss and test loss over epochs.
    train_acc = history.history['accuracy']
    test_acc = history.history['val_accuracy']

    # Set figure size.
    plt.figure(figsize=(12, 8))

    # Generate line plot of training, testing loss over epochs.
    plt.plot(train_acc, label='Training Accuracy', color='#185fad')
    plt.plot(test_acc, label='Validation Accuracy', color='orange')

    # Set title
    # plt.title('Training and Validation Accuracy by Epoch (15)', fontsize = 25)
    plt.xlabel('Epoch', fontsize = 18)
    plt.ylabel('Categorical Crossentropy', fontsize = 18)
    plt.xticks(range(15))

    plt.legend(fontsize = 18, loc="lower right");

    # Predict
    test_generator.reset()
    pred=model.predict(test_generator,verbose=1)

    predicted_class_indices=np.argmax(pred,axis=1)
    print(predicted_class_indices)

    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]
    filenames=test_generator.filenames
    print(filenames)
    print(predictions)

def predict_class(dest_path, image_, cs):
    img = tf.keras.utils.load_img(image_, target_size=(56, 56))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    model = load_model(
        get_py_path() + 'wi_vision/WI_required/color_model')
    predictions = model.predict(img_array, verbose = 0)
    predicted_class_indices=int(np.argmax(predictions,axis=1))
    x = color_cls[predicted_class_indices]
    if cs == 'False':
        print(x)


    elif cs == 'True':
      for c in color_cls:
        if c not in os.listdir(dest_path):
          os.mkdir(dest_path + c + '/')
      copy_to_folder([image_], dest_path + x + '/')
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', help='Provide image path here to classify the image', required = True)
    parser.add_argument('--dest_path', help='Provide path to store images into crushed uncrushed folders', required = True)
    parser.add_argument('--crop_save', help='True if crops are already saved.', default= 'False', type=str)
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