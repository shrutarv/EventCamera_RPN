# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 09:58:36 2023

@author: Richard
"""
import os
import cv2
from Detector2 import *
import json
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
from stoppingtime import *
import time

def load_images_from_folder(path,step):
    images = []
    lst = os.listdir(path)
    for i in range(0,len(lst),step):
        #print (i )
        img = cv2.imread((path+str(int(i))+".png"))
        
        if img is not None:
            images.append(img)
    return images

if __name__=="__main__":
    number_message_contatenate=10
    images=load_images_from_folder("D:/Masterarbeit_Experiment/Hall3/images/",number_message_contatenate)
    
    for i in range(len(images)):
        imageRGB=images[i]
        
        object_detector=Detector2(model_type="IS")
        prediction=object_detector.onImage(imageRGB )
        stoppingtime.end_time.append(time.time())
    stoppingtime.needed_time()