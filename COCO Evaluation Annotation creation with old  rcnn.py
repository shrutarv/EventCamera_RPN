# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 18:47:37 2023

@author: Richard
"""

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
import pycocotools.mask as maskUtils

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
    images=load_images_from_folder("D:/Masterarbeit_Experiment/Outside3/images/",number_message_contatenate)
    classes_to_include=[2]
    predicted_annotations1=[]
    image_id=0
    detection_id=0
    predictions_json1="D:/Masterarbeit_Experiment/Outside3/Outside3_oldRCNN_output.json"
    for j in range(len(images)):
        imageRGB=images[j]
        
        object_detector=Detector2(model_type="IS")
        prediction=object_detector.onImage(imageRGB)
        
        for i in range(len(prediction["instances"])):
            pred_class = prediction["instances"].__getattr__("pred_classes").tolist()[i]
            if pred_class in classes_to_include: # check if class is in the list of classes to include
            # extract other prediction information and append to the list
                print (f"durchläufe {i}")
                file_name=str(j*number_message_contatenate)+".png"
                print(file_name)
                if i>0:
                    print("Hallejuja")
                    abbruch=True
                
    
                mask = prediction["instances"].__getattr__("pred_masks")[0].tolist()
                mask = np.array(mask, order="F")
                prediction_mask = maskUtils.encode(mask)
                prediction_mask_counts=prediction_mask["counts"].decode()
                
    
                
                
               
                bBox=prediction["instances"].__getattr__("pred_boxes")
                pred_class=prediction["instances"].__getattr__("pred_classes").tolist()[i]
                score=prediction["instances"].__getattr__("scores").tolist()[i]
                bBox=bBox.tensor.tolist()
                bBox=bBox[i]
                bBox=[bBox[0],bBox[1],abs(bBox[2]-bBox[0]),abs(bBox[3]-bBox[1])]
                print(len(bBox))
                print(bBox)
                predicted_annotation1={"image_id":image_id,"file_name": file_name,"id":detection_id,"category_id":pred_class, "bbox":bBox,
                    "score":score,                  
                    "segmentation":{
                        "size":prediction_mask["size"],
                        "counts":prediction_mask_counts}
                    }
                #"bbox":{"0":bBox[i][0],"1":bBox[i][1],"2":bBox[i][2],"3":bBox[i][3]}
                
                
                predicted_annotations1.append(predicted_annotation1)
               
                detection_id+=1
        image_id+=1
with open(predictions_json1, 'w') as f:
    json.dump(predicted_annotations1 , f)    