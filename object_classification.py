# importing Libraries
import csv
from socket import PACKET_MULTICAST
from turtle import color
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import shutil
import argparse
import time
import multiprocessing
from requests import get
from torch import true_divide
# %matplotlib inline
# importing Dataset from drive
#from google.colab import drive
# drive.mount('/content/drive')

# <---TO CREATE FOLDERS (ONE TIME RUN)--->
# os.mkdir('/content/drive/MyDrive/object_detection_dataset/')
# os.mkdir('/content/drive/MyDrive/object_detection_dataset/test/')
# os.mkdir('/content/drive/MyDrive/object_detection_dataset/train/')
# os.mkdir('/content/drive/MyDrive/object_detection_dataset/val/')

# <---TO COPY IMAGES INTO DIRECTORY WE ARE GOING TO USE--->
"""
def copy_to_folder(img, destination_folder):
  for f in img:
        try:
            shutil.copy(f, destination_folder)
        except:
            print(f)
            assert False

src_path = '/content/drive/MyDrive/final_data/images/'
"""
# DIRECTORY LOCATION
#dest_path = '/content/drive/MyDrive/object_detection_dataset/'
#ttv = ['test/', 'train/','val/']
"""
for x in ttv:
  images = []
  for f in os.listdir(src_path + x):
    images.append(src_path + x + f)
  images.sort()
  copy_to_folder(images,dest_path + x)
"""

counter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
w_class = {0: 'pet_bottles', 1: 'plastic_wrapper', 2: 'ldpe_wrapper', 3: 'hdpe_bottle', 4: 'paper', 5: 'pp', 6: 'aluminium_foil',
               7: 'multilayer_plastic', 8: 'ps', 9: 'cardboard', 10: 'blister_pack', 11: 'aluminium_can', 12: 'tetrapack'}


def is_image(img_):
    if img_.split('.')[-1] in ['png', 'jpg', 'jpeg', 'raw']:
        return True
    else:
        return False

def get_py_path():
    return str(os.getcwd()) + '/'

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src_path', help='Provide Source Path of the Images', required=True, type=str)
    parser.add_argument(
        '--weights', default=get_py_path() + 'wi_vision/WI_required/object_weight_file/object_weight.pt', help='Path of the weight file.', type=str)
    parser.add_argument(
        '--dest_path', default= get_py_path() + 'wi_vision/WI_Folder/', help='Provide Destination Path to store the Output of Classificated Images.', type=str)
    parser.add_argument('--save_crops', help = 'To save Crop images from image', default= False, type = bool)
    parser.add_argument('--pet_class', help='To do pet classification', default= False, type = bool)
    parser.add_argument('--pet_loc', help='Provide path to save Pet Classificated images', default=get_py_path() + 'wi_vision/WI_Folder/')
    parser.add_argument('--color_class', help='To do color classification', default= False, type=bool)
    parser.add_argument('--color_loc', help='Provide path to save Pet Classificated images', default=get_py_path() + 'wi_vision/WI_Folder/')
    parser.add_argument('--save_csv', help='To save data into CSV file', default= True, type = bool)
    parser.add_argument('--csv_loc', default=get_py_path() + 'wi_vision/WI_Folder/', help= 'Provide path to save CSV file to.', type = str)
    args = parser.parse_args()

    try:
        if os.path.isdir(args.src_path):
            images_ = args.src_path
        elif is_image(args.src_path):
            images_ = args.src_path
        else:
            raise Exception('Provide path is neither an image nor a Directory that has images of format JPG PNG JPEG RAW. ARGUMENT only accepts this format files or directory that has this type of files')
        print(images_)
    
    except Exception as e:
        print(e)
    
    try:
        if args.weights.split('.')[-1] in ['pt', 'pb', 'onnx', 'torchscript', 'xml', 'engine', 'mlmodel', 'tflite', 'hdf5']:
            weights = args.weights
        else:
            raise Exception('The provided file is not a weight file.')
    except Exception as e:
        print(e)
    
    try:
        dest_path = args.dest_path
        
        if dest_path[-1] != '/':
            d_path = dest_path + '/'
        elif dest_path[-1] == '/':
                d_path = dest_path
        
        if 'crops' not in os.listdir(dest_path):
            if dest_path[-1] != '/':
                os.mkdir(dest_path + '/crops/')
                c_path = dest_path + '/crops/'
            elif dest_path[-1] == '/':
                os.mkdir(dest_path + 'crops/')
                c_path = dest_path + 'crops/'
        else:
            if dest_path[-1] != '/':
                c_path = dest_path + '/crops/'
            elif dest_path[-1] == '/':
                c_path = dest_path + '/crops/'
    except Exception as e:
        print(e)

    save_crops = args.save_crops
    pet_class = args.pet_class
    pet_loc = args.pet_loc
    color_class = args.color_class
    color_loc = args.color_loc
    save_csv = args.save_csv
    csv_loc = args.csv_loc
        
    return [images_, weights, d_path, c_path, save_crops, pet_class, pet_loc, color_class, color_loc, save_csv, csv_loc]
        
        
def yolov5_classifier(images_, weights, d_path):
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    os.system('git clone https://github.com/ultralytics/yolov5')
    req_run = 'pip install -r ' + get_py_path() + 'yolov5/requirements.txt'
    os.system(req_run)
    os_run = 'python3 ' + get_py_path() + 'yolov5/detect.py --data ' + get_py_path() + 'yolov5/data/coco128.yaml --source ' + \
        str(images_) + ' --weights ' + str(weights) + \
        ' --conf 0.25  --save-txt --nosave --project ' + str(d_path)
    
    os.system(os_run)

def image_processing(images_, d_path, c_path, sc, pc, pl, cc, cl, s_csv, cs):
    src_path = images_  # Source Image Path from where to detect objects
    label_dir = d_path + 'exp/labels/'  # Label Directory Path
    crop_store = c_path  # Save Path
    #counter_store = '/content/drive/MyDrive/object_detection_dataset/exp3/counter/'
    img_id = 0
    
    counter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    w_class = {0: 'pet_bottles', 1: 'plastic_wrapper', 2: 'ldpe_wrapper', 3: 'hdpe_bottle', 4: 'paper', 5: 'pp', 6: 'aluminium_foil',
               7: 'multilayer_plastic', 8: 'ps', 9: 'cardboard', 10: 'blister_pack', 11: 'aluminium_can', 12: 'tetrapack'}
    
    label_path_arr = []
    img_arr = []
    for w_c in w_class.keys():
        if w_class[w_c] not in os.listdir(crop_store):
            os.mkdir(crop_store + w_class[w_c] + '/')
    
    for t in os.listdir(label_dir):
        label_path_arr.append(t)
    if is_image(src_path):
        img_arr.append(src_path)
    else:
        for i in os.listdir(src_path):
            img_arr.append(i)
        
    for f in img_arr:
        csv_row = ['', '', -1, '', [-1, -1, -1, -1], -1, '']
        x = f.split(".")
        label_path = x[0]+".txt"
        if label_path in label_path_arr:
            img = cv2.imread(src_path + f)
            csv_row[0] = src_path + f
            h, w, _ = img.shape
            #img_ = cv2.resize(img, [int(h), int(w/2)])
            with open(label_dir + label_path) as file_:
                lines = file_.readlines()
                for line in lines:
                    i, x_cen, y_cen, wi, hi = line.split(' ')
                    i = int(i)
                    x_cen = int(float(x_cen) * w)
                    y_cen = int(float(y_cen) * h)
                    wi = int(float(wi) * w)
                    hi = int(float(hi) * h)

                    counter[i] = counter[i] + 1
                    csv_row[1] = w_class[i]
                    csv_row[2] = i
                    csv_row[4][0], csv_row[4][1], csv_row[4][2], csv_row[4][3] = x_cen, y_cen, wi, hi
                    csv_row[5] = counter[i] 
                    csv_row[6] = ' '
                    if sc == True:                  
                        roi = img[(y_cen-int(hi/2)+5):(y_cen+int(hi/2)),
                                (x_cen-int(wi/2)+5):(x_cen+int(wi/2))]
                        cv2.imwrite(
                            crop_store + w_class[i] + '/' + x[0] + '_' + str(i) + '_' + str(counter[i]) + '.png', roi)
                    
                    if pc == True:
                        if i == 0:
                            pet_call_process = multiprocessing.Process(target=call_pet, args=[str(crop_store + w_class[0] + '/' + x[0] + '_' + str(i) + '_' + str(counter[i]) + '.png')])
                            pet_call_process.start()
                            time.sleep(5)
                    
                    if cc == True:
                        pass
                    
                    if save_csv == True:
                        pass


                file_.close()

def call_pet(src_path):
    os_pet_command = 'python3 pet_classification.py --src_path ' + src_path
    os.system(os_pet_command)

if __name__ == "__main__":
    os.system('git clone https://github.com/Henishwi/wi_vision')
    images_, weights, d_path, c_path, save_crops, pet_class, pet_loc, color_class, color_loc, save_csv, csv_loc  = parse_opt()
    yolov5_classifier(images_, weights, d_path)
    image_processing(images_, d_path, c_path, save_crops, pet_class, pet_loc, color_class, color_loc, save_csv, csv_loc)