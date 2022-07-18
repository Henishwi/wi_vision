import os
import cv2
from cv2 import destroyAllWindows
import numpy as np
import argparse
from numpy.linalg import norm
from common import *
from keras.models import load_model
import tensorflow as tf
from tensorflow import keras

name = '/home/henishv5/final_dataset/frame_149'

def bb_intersection_over_union(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou


def parse_opt():
  parser = argparse.ArgumentParser()
  parser.add_argument('--src_path', help='Enter path to your image or image directory')
  parser.add_argument('--dest_path', help='Enter path directory to where Unknow Images and Annotation Files to store')
  parser.add_argument('--annotation', help='Enter path of Annotation File.')
  args = parser.parse_args()
  try:
        if os.path.isdir(args.src_path):
            images_ = args.src_path
        elif is_image(args.src_path):
            images_ = args.src_path
        elif is_bag_file(args.src_path):
            print("Process BagFile")
            images_ = process_bagfiles(args.src_path, args.topic_name)
        else:
            raise Exception('Provide path is neither an image nor a Directory that has images of format JPG PNG JPEG RAW. ARGUMENT only accepts this format files or directory that has this type of files')
        print(images_)
  except Exception as e:
      print(e)
  
  try:
        dest_path = args.dest_path
        if 'unknown' not in os.listdir(dest_path):
            os.mkdir(dest_path + 'unknown/')
            c_path = dest_path + 'unknown/'
        else:
            c_path = dest_path + 'unknown/'
  except Exception as e:
      print(e)

  
  annot_path = args.annotation
  
  return images_, c_path, annot_path

def u_obj(src_path, dest_path, a_path):
  image_lst = []
  if os.path.isdir(src_path):
      for f in os.listdir(src_path):
        image_lst.append(src_path + f)
  elif is_image(src_path):
        image_lst.append(src_path)
  else:
    raise Exception("File format not legal.")

  for name in image_lst:
    img_ = cv2.imread(name)
    h, w, _ = img_.shape
    img_ = img_[:, int(w*0.10):int(w-(w*0.10))]

    x = np.average(norm(img_, axis=2)) / np.sqrt(3)

    if x < 70:
      i = 2 * (cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    elif x > 150:
      i = 1/2 * cv2.THRESH_OTSU - cv2.THRESH_BINARY_INV
    else:
      i = cv2.THRESH_BINARY

    img_gray_ = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray_, 110, 255, int(i))

    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    try: hierarchy = hierarchy[0]
    except: hierarchy = []
    h, w, _ = img_.shape
    lst = []
    annot = []
    # computes the bounding box for the contour, and draws it on the frame,
    for contour, hier in zip(contours, hierarchy):
        (x,y,wi,hi) = cv2.boundingRect(contour)
        if wi > 100 and hi > 100 and wi < 900 and hi <900:
            cv2.rectangle(thresh, (x,y), (x+wi,y+hi), (255, 0, 0), 2)
            lst.append([x,y,x+wi,y+hi])
            annot.append([15, (x+wi/2)/w, (y+hi/2)/h, wi/w, hi/h])

    lst_check = [0 for i in range(len(lst))]

    og_lst = []
    with open(a_path + str(name.split('/')[-1]).split('.')[0] + '.txt', encoding = 'utf-8') as f:
      lines = f.readlines()
      for line in lines:
          temp_ = []
          wi = int(float(line.split(" ")[3]) * w) + 2
          hi = int(float(line.split(" ")[4]) * h) + 2
          x1 = int(float(line.split(" ")[1]) * w) - int(wi/2) + 2
          y1 = int(float(line.split(" ")[2]) * w) - int(hi/2) + 2
          x2 = x1+wi
          y2 = y1+hi
          temp_.append(x1)
          temp_.append(y1)
          temp_.append(x2)
          temp_.append(y2)
          og_lst.append(temp_)

    i = 0
    for t in lst:
      lt = -1
      for u in og_lst:
        x0, y0, x1, y1 = t
        x2, y2, x3, y3 = u
        iou = bb_intersection_over_union([x0, y0, x1, x1], [x2, y2, x3, y3])
        if iou *100 > 5:
          lt = 1
          break
        else:
          lt = 0
      if lt == 1:
        lst_check[i] = 1
      else:
        lst_check[i] = 0
      i += 1
    j = 0
    
    for ls in range(len(lst_check)):
      if lst_check[ls] != 1:
        x1 = lst[ls][0]
        y1 = lst[ls][1]
        x2 = lst[ls][2]
        y2 = lst[ls][3]
        
        """if predictions[0][0] == 1:
            x = pet_classes[0]
        else:
            x = pet_classes[1]"""
        with open(dest_path + name.split(".")[0].split("/")[-1] + '.txt', 'a+') as f:
          
          #print(str(annot[ls][0]) + ' ' + str(annot[ls][1]) + ' ' + str(annot[ls][2]) + ' ' + str(annot[ls][3]) + ' ' + str(annot[ls][4]))
          store_path = dest_path + str(name.split(".")[0]).split("/")[-1] + '_' + str(j) + '.jpg'
          j += 1
          cv2.imwrite(store_path, img_[y1:y2, x1:x2])
          img = tf.keras.utils.load_img(store_path, target_size=(56, 56))
          img_array = tf.keras.utils.img_to_array(img)
          img_array = tf.expand_dims(img_array, 0) # Create a batch
                
          model = load_model('/home/henishv5/Downloads/400_w/best.pt')
          predictions = model.predict(img_array, verbose = 0)
          print(predictions)
          break
          f.writelines(str(annot[ls][0]) + ' ' + str(annot[ls][1]) + ' ' + str(annot[ls][2]) + ' ' + str(annot[ls][3]) + ' ' + str(annot[ls][4])+ '\n')
        
      else:
        pass
  



if __name__ == '__main__':
  s_path, d_path, a_path = parse_opt()
  u_obj(s_path, d_path, a_path)