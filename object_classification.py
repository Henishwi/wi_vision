# importing Libraries
from concurrent.futures import process
import csv
from email import parser
from socket import PACKET_MULTICAST
from matplotlib import image
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
import os
import shutil
import argparse
import rospy
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import argparse
import time
import multiprocessing
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

def is_bag_file(bag_path):
    if bag_path.split('.')[-1] == 'bag':
        print(True)
        return True
    else:
        print(False)
        return False

def get_py_path():
    return str(os.getcwd()) + '/'



def process_bagfiles(bag_path, topic_name):
    output_dir = get_py_path() + 'wi_vision/WI_Folder/Bag_Images/'
    bag_file = bag_path
    image_topic = topic_name
    if 'Bag_Images' not in os.listdir(get_py_path() + 'wi_vision/WI_Folder/'):
            os.mkdir(output_dir)
    count = len(os.listdir(output_dir)) + 1
    
    bag = rosbag.Bag(bag_file, "r")
    bridge = CvBridge()
    for topic, msg, t in bag.read_messages(topics=[image_topic]):
        cv_img = bridge.imgmsg_to_cv2(msg , desired_encoding="bgr8")    
        cv2.imwrite(os.path.join(output_dir, "frame_" + str(count) + ".jpg"), cv_img)
        count += 1
    
    
    return output_dir

    
    

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src_path', help='Provide Source Path of the Images', required=True, type=str)
    parser.add_argument(
        '--weights', default=get_py_path() + 'wi_vision/WI_required/object_weight_file/object_weight.pt', help='Path of the weight file.', type=str)
    parser.add_argument(
        '--dest_path', default= get_py_path() + 'wi_vision/WI_Folder/', help='Provide Destination Path to store the Output of Classificated Images.', type=str)
    parser.add_argument('--save_crops', help = 'To save Crop images from image', default= False, type = bool)
    parser.add_argument('--topic_name', default = '/pylon_camera_node/image_rect_color')
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
        elif is_bag_file(args.src_path):
            print("Process BagFile")
            images_ = process_bagfiles(args.src_path, args.topic_name)
            print(images_)
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
        ' --conf 0.25  --save-txt --nosave --project ' + str(d_path) + ' --name exp --exist-ok'
    
    os.system(os_run)

def image_processing(images_, d_path, c_path, sc, pc, pl, cc, cl, s_csv, cs):
    src_path = images_  # Source Image Path from where to detect objects
    label_dir = d_path + 'exp/labels/'  # Label Directory Path
    crop_store = c_path  # Save Path
    #counter_store = '/content/drive/MyDrive/object_detection_dataset/exp3/counter/'
    
    counter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    w_class = {0: 'pet', 1: 'ldpe_wrapper', 2: 'hdpe', 3: 'paper', 4: 'pp', 5: 'aluminium_foil',
               6: 'multilayer_plastic', 7: 'cardboard', 8: 'aluminium_can', 9: 'tetrapack'}
    
    label_path_arr = []
    img_arr = []
    for w_c in w_class.keys():
        if w_class[w_c] not in os.listdir(crop_store):
            os.mkdir(crop_store + w_class[w_c] + '/')
        else:
            counter[w_c] = len(os.listdir(crop_store + w_class[w_c] + '/'))
    
    for t in os.listdir(label_dir):
        label_path_arr.append(t)
    if is_image(src_path):
        img_arr.append(src_path)
    else:
        for i in os.listdir(src_path):
            img_arr.append(i)
        
    for f in img_arr:
        csv_row = ['', '', -1, ' ', [-1, -1, -1, -1], -1, '']
        x = f.split(".")
        label_path = x[0]+".txt"
        i, x_cen, y_cen, wi, hi = 0,0,0,0,0
        img = cv2.imread(src_path + f)
        if label_path in label_path_arr:
            csv_row[0] = src_path + f
            h, w, _ = img.shape
            img_ = cv2.resize(img, [int(h), int(w/2)])
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
                    
                    roi = img[(y_cen-int(hi/2)+5):(y_cen+int(hi/2)), (x_cen-int(wi/2)+5):(x_cen+int(wi/2))]
                    #img_copy = img.copy()
                    #cv_img = cv2.rectangle(img_copy, ((x_cen+int(wi/2)),(y_cen+int(hi/2))), ((x_cen-int(wi/2)+5), (y_cen-int(hi/2)+5)), color = (0,255,0))
                    #cv2.imshow('Image', cv_img)
                    #if cv2.waitKey(1) == ord('q'):
                    #    break

                    if sc == True:                 
                        cv2.imwrite(
                            crop_store + w_class[i] + '/' + x[0] + '_' + str(i) + '_' + str(counter[i]) + '.jpg', roi)
                        if cc == True:
                            time.sleep(10)
                            color_call_process = multiprocessing.Process(target=call_color, args=[str(crop_store + w_class[i] + '/'+ x[0] + '_' + str(i) + '_' + str(counter[i]) + '.jpg'), cl, sc])
                            color_call_process.start()
                            

                    else:
                        if i == 0:
                            cv2.imwrite(crop_store + w_class[i] + '/' + x[0] + '_' + str(i) + '_' + str(counter[i]) + '.jpg', roi)
                            for w_c in w_class.keys():
                                if w_class[w_c] in os.listdir(crop_store):
                                    if len(os.listdir(crop_store + w_class[w_c] + '/')) == 0 and w_c!=0:
                                        os.rmdir(crop_store + w_class[w_c] + '/')
                                else:
                                    pass
                        
                    if pc == True:
                        if i == 0:
                            pet_call_process = multiprocessing.Process(target=call_pet, args=[str(crop_store + w_class[0] + '/' + x[0] + '_' + str(i) + '_' + str(counter[i]) + '.png'), pl, sc])
                            pet_call_process.start()
                            time.sleep(5)
                    
                    
                    if save_csv == True:
                        with open(get_py_path() + 'waste_data.csv', 'a+', encoding='UTF8', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(csv_row)
            file_.close()
        """if wi == 0 and hi == 0:
            cv2.imshow("Image", img)
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            x_cen = int(float(x_cen) * w)
            y_cen = int(float(y_cen) * h)
            wi = int(float(wi) * w)
            hi = int(float(hi) * h)
            img_copy = img.copy()
            cv_img = cv2.rectangle(img_copy, ((x_cen+int(wi/2)),(y_cen+int(hi/2))), ((x_cen-int(wi/2)+5), (y_cen-int(hi/2)+5)), color = (0,255,0), thickness=2)
            cv2.imshow('Image', cv_img)
            if cv2.waitKey(1) == ord('q'):
                break"""
        
        #time.sleep(0.3)

    #cv2.destroyAllWindows()

                

    

def call_color(src_path, dest_path, sc):
    os_pet_command = 'python3 color_classification.py --src_path ' + src_path + ' --dest_path ' + dest_path + ' --crop_save ' + str(sc)
    os.system(os_pet_command)

def call_pet(src_path, dest_path, sc):
    os_pet_command = 'python3 pet_classification.py --src_path ' + src_path + ' --dest_path ' + dest_path + ' --crop_save ' + str(sc)
    os.system(os_pet_command)

if __name__ == "__main__":
    os.system('git clone https://github.com/Henishwi/wi_vision')
    images_, weights, d_path, c_path, save_crops, pet_class, pet_loc, color_class, color_loc, save_csv, csv_loc  = parse_opt()
    yolov5_classifier(images_, weights, d_path)
    if 'yolov5' in os.listdir(get_py_path()):
        image_processing(images_, d_path, c_path, save_crops, pet_class, pet_loc, color_class, color_loc, save_csv, csv_loc)