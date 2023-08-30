# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 12:07:51 2023

@author: Richard
"""
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import numpy as np
from coordinates import *


class Detector2:
    def __init__(self, model_type="OD" ):
        
        
        """"
        coor_x_min=400.6273
        coor_x_max=624.1776
        coor_y_min=388.1294
        coor_y_max=756.6057
        
        coordinates.coor.append([coor_x_min,coor_y_min,coor_x_max,coor_y_max])
        
        
        print(coordinates.coor)
        """ 
        self.cfg=get_cfg()
        self.model_type=model_type
        
        #self.cfg.set_coor(coor)
        if model_type=="OD":
        
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS=model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        
        elif model_type=="IS":
            
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS=model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
       
        elif model_type=="KP":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS=model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        
        elif model_type=="LVIS":
            self.cfg.merge_from_file(model_zoo.get_config_file("LIVSv0.5-IinstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
            self.cfg.MODEL.WEIGHTS=model_zoo.get_checkpoint_url("LIVSv0.5-IinstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")
        
        elif model_type=="PS":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
            self.cfg.MODEL.WEIGHTS=model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    
       
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.05
        self.cfg.MODEL.DEVICE="cuda"
        
        self.predictor=DefaultPredictor(self.cfg)
        
    def onImage (self,image):
        #image =cv2.imread(imagePath)
        #image=image[20:460, 20:550]
        
        if self.model_type !="PS":
            
            height = 480
            width = 640
            dim = (width, height)
            image=cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
            #print (dim)
            #dim =(1333,773)
            
            
            
            predictions =self.predictor(image)
            #print(f"predictions: {predictions}")
            
            viz=Visualizer(image[:,:,::-1], metadata = MetadataCatalog.get( self.cfg.DATASETS.TRAIN[0]),instance_mode=ColorMode.IMAGE_BW,scale=0.5)
            output=viz.draw_instance_predictions(predictions["instances"].to("cpu"))
            
        else:
            predictions,segmentInfo=self.predictor(image)["panoptic_seg"]
            print("inside predictions")
            viz = Visualizer(image[:,:,::-1],MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
            output=viz.draw_panoptic_seg(predictions.to("cpu"), segmentInfo)
            # image=output.get_image()
          
        cv2.imshow("Result", output.get_image()[:,:,::-1])
        
        cv2.waitKey(5)
        #cv2.destroyAllWindows()
        return predictions
       
    def onVideo(self, videoPath):
        cap=cv2.VideoCapture(videoPath)
        
        if (cap.isOpened()==False):
            print("Error open")
            return
        (sucess,image)=cap.read()
        
        while sucess:
            if self.model_type!="PS":
                height = int(image.shape[0]*0.5)
                width = int(image.shape[1]*0.5)
                dim = (width, height)
                image=cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
                predictions =self.predictor(image)
                
                viz=Visualizer(image[:,:,::-1], metadata = MetadataCatalog.get( self.cfg.DATASETS.TRAIN[0]),instance_mode=ColorMode.IMAGE_BW)
                output=viz.draw_instance_predictions(predictions["instances"].to("cpu"))
                
            else:
                predictions,segmentInfo=self.predictor(image)["panoptic_seg"]
                viz = Visualizer(image[:,:,::-1],MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
                output=viz.draw_panoptic_seg(predictions.to("cpu"), segmentInfo)

            print("show image")
            cv2.imshow("Result", output.get_image()[:,:,::-1])
            #key=cv2.waitKey(1) & 0xFF
            #if key==ord("q"):
            #   break
            (sucess,image)=cap.read()
            