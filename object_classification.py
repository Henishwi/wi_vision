# importing Libraries
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import shutil
import argparse
import time
import multiprocessing
# %matplotlib inline
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.system('git clone https://github.com/ultralytics/yolov5')
os.system('pip install -r /home/henishv5/WI_Testing/yolov5/requirements.txt')
os.system('cd ..')
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

#!git clone https://github.com/ultralytics/yolov5


def is_image(img_):
    if img_.split('.')[-1] in ['png', 'jpg', 'jpeg', 'raw']:
        return True
    else:
        return False

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src_path', help='Provide Source Path of the Images', required=True, type=str)
    parser.add_argument(
        '--weights', default='/home/henishv5/WI_Testing/yolov5s_v1/40_ep/weights/best.pt', type=str)
    parser.add_argument(
        '--dest_path', help='Provide Destination Path to store the Output of Classificated Images.', required=True, type=str)
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
        if args.weights.split('.')[-1] in ['pt', 'pb', 'onnx', 'torchscript', 'xml', 'engine', 'mlmodel', 'tflite']:
            weights = args.weights
        else:
            raise Exception('The provided file is not a weight file.')
    except Exception as e:
        print(e)
    
    try:
        dest_path = args.dest_path
        if 'data_temp' not in os.listdir(dest_path):
            if dest_path[-1] != '/':
                os.mkdir(dest_path + '/data_temp/')
                d_path = dest_path + '/data_temp/'
            elif dest_path[-1] == '/':
                os.mkdir(dest_path + 'data_temp/')
                d_path = dest_path + 'data_temp/'
    
        else:
            if dest_path[-1] != '/':
                d_path = dest_path + '/data_temp/'
            elif dest_path[-1] == '/':
                d_path = dest_path + 'data_temp/'
    
        if'data_final' not in os.listdir(dest_path):
            if dest_path[-1] != '/':
                os.mkdir(dest_path + '/data_final/')
            elif dest_path[-1] == '/':
                os.mkdir(dest_path + 'data_final/')
        if 'crops' not in os.listdir(dest_path + '/data_final/'):
            if dest_path[-1] != '/':
                os.mkdir(dest_path + '/data_final/crops/')
                c_path = dest_path + '/data_final/crops/'
            elif dest_path[-1] == '/':
                os.mkdir(dest_path + '/data_final/crops/')
                c_path = dest_path + '/data_final/crops/'
        else:
            if dest_path[-1] != '/':
                c_path = dest_path + '/data_final/crops/'
            elif dest_path[-1] == '/':
                c_path = dest_path + 'data_final/crops/'
    
    
    except Exception as e:
        print(e)
        
    return [images_, weights, d_path, c_path]
        
        
def yolov5_classifier(images_, weights, d_path):
    os_run = 'python3 /home/henishv5/WI_Testing/yolov5/detect.py --data /home/henishv5/WI_Testing/yolov5/data/coco128.yaml --source ' + \
        str(images_) + ' --weights ' + str(weights) + \
        ' --conf 0.25  --save-txt --nosave --project ' + str(d_path)
    
    os.system(os_run)

def image_classification_into_folders(images_, d_path, c_path):
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
        x = f.split(".")
        label_path = x[0]+".txt"
        if label_path in label_path_arr:
            img = cv2.imread(src_path + f)
            h, w, _ = img.shape
            #img_ = cv2.resize(img, [int(h), int(w/2)])
            with open(label_dir + label_path) as file_:
                lines = file_.readlines()
                for line in lines:
                    i, x_cen, y_cen, wi, hi = line.split(' ')
                    x_cen = int(float(x_cen) * w)
                    y_cen = int(float(y_cen) * h)
                    wi = int(float(wi) * w)
                    hi = int(float(hi) * h)
                    i = int(i)
                    roi = img[(y_cen-int(hi/2)+5):(y_cen+int(hi/2)),
                              (x_cen-int(wi/2)+5):(x_cen+int(wi/2))]
                    counter[i] = counter[i] + 1
                    
                    cv2.imwrite(
                        crop_store + w_class[i] + '/' + x[0] + '_' + str(i) + '_' + str(counter[i]) + '.png', roi)
                    if i == 0:
                        pet_call_process = multiprocessing.Process(target=call_pet, args=[str(crop_store + w_class[i] + '/' + x[0] + '_' + str(i) + '_' + str(counter[i]) + '.png')])
                        pet_call_process.start()
                        time.sleep(5)
                file_.close()

def call_pet(src_path):
    os_pet_command = 'python3 pet_classification.py --src_path ' + src_path
    os.system(os_pet_command)
                
if __name__ == "__main__":
    i, w, d, c = parse_opt()
    yolov5_classifier(i, w, d)
    image_classification_into_folders(i, d, c)