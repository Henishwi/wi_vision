# Importing Libraries
import os
import shutil
import cv2
import rosbag
from cv_bridge import CvBridge

def is_image(img_):
    try:
        if img_.split('.')[-1] in ['png', 'jpg', 'jpeg', 'raw']:
            return True
        else:
            return False
    except Exception as e:
        print("is_image() {}".format(e))

def is_bag_file(bag_path):
    try:
        if bag_path.split('.')[-1] == 'bag':
            return True
        else:
            return False
    except Exception as e:
        print("is_bag_file() {}".format(e))

def get_py_path():
    try:
        return str(os.getcwd() + '/')
    except Exception as e:
        print("get_py_path() {}".format(e))

def process_bagfiles(bag_path, topic_name):
    try:
        output_dir = get_py_path() + 'wi_required/Bag_Images/'
        bag_file = bag_path
        image_topic = topic_name
        if 'Bag_Images' not in os.listdir(get_py_path() + 'wi_required/'):
                os.mkdir(output_dir)
        count = len(os.listdir(output_dir)) + 1
        
        bag = rosbag.Bag(bag_file, "r")
        bridge = CvBridge()
        for topic, msg, t in bag.read_messages(topics=[image_topic]):
            cv_img = bridge.imgmsg_to_cv2(msg , desired_encoding="bgr8")    
            cv2.imwrite(os.path.join(output_dir, "frame_" + str(count) + ".jpg"), cv_img)
            count += 1
        return output_dir
    except Exception as e:
        print("process_bagfiles() {}".format(e))

def move_to_folder(file_, dest_path):
    try:
        shutil.move(file_, dest_path)
    except Exception as e:
        print("move_to_folder() {}".format(e))

def copy_to_folder(list_of_files, dest_path):
    for f in list_of_files:
        try:
            shutil.copy(f, dest_path)
        except Exception as e:
            print("copy_to_folder() {}".format(e))