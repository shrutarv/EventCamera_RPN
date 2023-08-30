# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 12:05:16 2023

@author: Richard
"""



import rosbag
#import bagpy
import pandas as pd
import numpy as np
import cv2
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import DBSCAN
#import datetime as dt
import time
import os

from coordinates import *
import cv2
from Detector2 import *
import json
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

#ROOT_DIR = 'D:/Masterarbeit_Experiment/extraction_scripts3'

#BAGFILE = "D:/Masterarbeit_Experiment/extraction_scripts2/events.bag"
#BAGFILE = "D:/Masterarbeit_Experiment/extraction_scripts3/events.bag"
#BAGFILE = "D:/Masterarbeit_Experiment/bridge2/bridge2_event.bag"
#BAGFILE = "D:/Masterarbeit_Experiment/Hall4/events_hall4.bag"
BAGFILE="D:/Masterarbeit_Experiment/Hall4/events_hall4.bag"
def visualise(clusters,xs,ys, image_f):
    color = []
    color.append('yellow')
    color.append('red')
    color.append('green')
    color.append('blue')
    color.append('orange')
    color.append('pink')
    color.append('cyan')
    color.append('orange')
    color.append('brown')
    color.append('white')
    color.append('lime')
    color.append('magenta')
    color.append('gray')
    color.append('purple')
    
    
    
    plt.figure()
    #plt.imshow(image)
    plt.imshow(image_f)
    for i in range(0,len(clusters)):
        m = i
        if m>13:
            m = 13
        plt.scatter(clusters[i][:,0], clusters[i][:,1],color = color[m],s=1)
        #plt.scatter(clusters[1][:,0], clusters[1][:,1],color = 'cyan',s=1)
        #plt.scatter(clusters[4][:,0], clusters[4][:,1],color = 'red', s=1)
        #plt.scatter(clusters[3][:,0], clusters[3][:,1],color = 'blue', s=1)
        #plt.scatter(clusters[2][:,0], clusters[2][:,1],color = 'green', s=1)
        #plt.scatter(clusters[6][:,0], clusters[6][:,1],color = 'green', s=1)
        #plt.scatter(clusters[5][:,0], clusters[5][:,1],color = 'brown', s=1)
        #plt.scatter(clusters[8][:,0], clusters[8][:,1],color = 'pink', s=1)
        #plt.scatter(clusters[9][:,0], clusters[9][:,1],color = 'pink', s=1)
        
        # Creating a convex hull around  objects
        #hull = ConvexHull(clusters[i])
        #plt.plot(clusters[i][:,0], clusters[i][:,1], 'o')
        #for simplex in hull.simplices:
         #   plt.plot(clusters[i][simplex, 0], clusters[i][simplex, 1], color = 'pink')
        
    plt.plot(xs,ys)
    # print(xs,ys)
    plt.show()
        
def min_max(clusters, bb_areas):
    x_mi = np.zeros(len(clusters))
    x_ma = np.zeros(len(clusters))
    y_mi = np.zeros(len(clusters))
    y_ma = np.zeros(len(clusters))
    m = 0
    for i in range(0,len(clusters)):
        x_min = np.min(clusters[i][:,0])
        x_max = np.max(clusters[i][:,0])
        y_min = np.min(clusters[i][:,1])
        y_max = np.max(clusters[i][:,1])
        area = (x_max - x_min)*(y_max - y_min)
        if area > 200:
            bb_areas[m] = area
            x_mi[m] = x_min
            x_ma[m] = x_max
            y_mi[m] = y_min
            y_ma[m] = y_max
            
            m += 1
        
        '''
        if np.min(clusters[i][:,0]) < x_min:
            x_min = np.min(clusters[i][:,0])
        if np.max(clusters[i][:,0]) > x_max:
            x_max = np.max(clusters[i][:,0])
        if np.min(clusters[i][:,1]) < y_min:
            y_min = np.min(clusters[i][:,1])
        if np.max(clusters[i][:,1]) > y_max:
            y_max = np.max(clusters[i][:,1])
          '''  
    return x_mi, x_ma, y_mi, y_ma, bb_areas

def clustering(data,e, ms):
    clusters = []
    # eps: The distance that specifies the neighborhoods. Two points are considered to be neighbors if the distance between them are less than or equal to eps.
    # min_samples(ms): Minimum number of data points to define a cluster.
    try:
        clustering = DBSCAN(
        eps=e,
        min_samples=ms,
        algorithm='kd_tree',
        n_jobs = 6,
        ).fit(data)

        # list of every cluster, including noise (assigned to -1).
        found_clusters = set(clustering.labels_)

        # remove noise
        found_clusters.remove(-1)

        # get each label class in it's own cluster collection
        for i, each in enumerate(found_clusters):
            points_with_class = np.column_stack(
                (data, clustering.labels_))

            detected_pixel_indexes = np.where(
                points_with_class[:, 2] == each)

            detected_pixels = points_with_class[detected_pixel_indexes]

            clusters.append(detected_pixels[:, 0:2])
            
       
        #visualise(clusters)
    except Exception as e: 
        # we allow this to fail gracefully if no detections are found
        print('Error:{}'.format(e))
    return clusters

def boundingBox(cluster,image_filtered):
    bb_areas = np.zeros(len(cluster))
    x_min, x_max, y_min, y_max, bb_areas = min_max(cluster, bb_areas)
    """
    x_min=x_min*0.8
    x_max=x_max*1.2  
    y_min=y_min*0.8
    y_max=y_max*1.2
    print(f"y_max= {y_max}")
    for i in range(len(x_min)):
        if x_max[i]>639:
            x_max[i]=639.0
            
        if y_max[i]>479:
            y_max[i]=479.0 
    """    
    coord = [[x_min,y_min], [x_min,y_max], [x_max,y_max], [x_max,y_min]]
    coord.append(coord[0])
    xs, ys = zip(*coord) #create lists of x and y values
    # uncomment to visualize
    visualise (cluster,xs,ys,image_filtered)
    #print(f"x_min:{x_min}  y_min:{y_min}      x_max:{x_max}  y_max:{y_max}")
    return   x_min, x_max, y_min*1.1, y_max*0.9
    
def segment(x, y, image):    
    kernel = np.ones((2, 1), np.uint8)
    
    image_filtered = cv2.erode(image, kernel, iterations=1)
    
    yx_coords = np.column_stack(np.where(image_filtered > 0))
    y_temp = yx_coords[:,0]
    x_temp = yx_coords[:,1]
    #plt.figure()
    #plt.imshow(image_filtered)
    #x_temp = x[p-ws:p]
    #y_temp = y[p-ws:p]
    #x_t = x_temp.reshape(ws,1)
    #y_t = y_temp.reshape(ws,1)
    x_t = x_temp.reshape(len(x_temp),1)
    y_t = y_temp.reshape(len(y_temp),1)
    data = np.concatenate((x_t,y_t),1)
    eps = 30
    ms = 50
    
    clusters = clustering(data, eps, ms)
    # Calculating bounding box coordinates
    #x_min, x_max, y_min, y_max = min_max(clusters)
    x_min, x_max, y_min, y_max=boundingBox(clusters, image_filtered)
    #coord = [[x_min,y_min], [x_min,y_max], [x_max,y_max], [x_max,y_min]]
    #coord.append(coord[0])
    #xs, ys = zip(*coord) #create lists of x and y values
    #visualise (clusters,xs,ys)
    image[:,:] = 0    
    return  [x_min ,y_min,x_max, y_max]
"""
def detector_init():
    cfg=get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS=model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.7
    cfg.MODEL.DEVICE="cpu"
    predictor=DefaultPredictor(cfg)

def detector_onImage(image):
    predictions,segmentInfo=predictor(image)["panoptic_seg"]
    viz = Visualizer(image[:,:,::-1],MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
    output=viz.draw_panoptic_seg(predictions.to("cpu"), segmentInfo)
    image=output.get_image()
    
    cv2.imshow("Result", output.get_image()[:,:,::-1])
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""

def load_images_from_folder(path,step):
    images = []
    lst = os.listdir(path)
    for i in range(0,len(lst),step):
        #print (i )
        img = cv2.imread((path+str(int(i))+".png"))
        
        if img is not None:
            images.append(img)
    return images



if __name__ == '__main__':
    bag = rosbag.Bag(BAGFILE)
    TOPIC = '/dvs/events'
    size = 400000
    x = np.zeros(size)   
    y = np.zeros(size)
    image = np.zeros((480,640))
    polarity = np.zeros(size)
    timestamp = np.zeros(size)
    count = 0
    num_msgs = 0
    segmentation_time=0
    time_detectron=0
    ws = 20000
    start_time = time.time()    
    event_message = bag.read_messages(TOPIC)
    #event_message = bag.message_by_topic(TOPIC)
    #num_msgs = bag.get_message_count(TOPIC) 
    number_message_contatenate = 10
    
    start_loading_frames=time.time()
    images=load_images_from_folder("D:/Masterarbeit_Experiment/Hall4/images/",number_message_contatenate)
    end_loading_frames=time.time()
    loading_frames_time=end_loading_frames-start_loading_frames
    
    
    num_picture=0
    
    
    
    for top,msg,tim in bag.read_messages(TOPIC):
        start_time_segmentation=time.time()
        #print("next message")
        
        num_msgs += 1
            
        #for message_traverser in range(num_msgs):
        for event_traverser in range(len(msg.events)):
            #timestamp.append((msg.events)[event_traverser].ts.secs)
            
            if(num_msgs%10 != 0 ): # 
                break
            
            x[count] = (msg.events)[event_traverser].x
            y[count] = (msg.events)[event_traverser].y
            polarity = (msg.events)[event_traverser].polarity
            image[(msg.events)[event_traverser].y,(msg.events)[event_traverser].x] = int((msg.events)[event_traverser].polarity)*254
            count += 1
            #print(event_traverser)
        if num_msgs%number_message_contatenate == 0 :
            #print("execute segmentation script")
            coor=segment(x,y, image)
            end_time_segmentation=time.time()
            segmentation_time+= (end_time_segmentation-start_time_segmentation)
            #print(coor)
            #print(coor)      
            
            start_time_identification=time.time()
            for i in range(len(coor[0])):
                coordinates.coor.append([coor[0][i],coor[1][i],coor[2][i],coor[3][i]])
            
            #print(coordinates.coor)
            x = np.zeros(size)
            y = np.zeros(size)
            
            #imageRGBPath="D:/Masterarbeit_Experiment/extraction_scripts/images/"+str(int(num_msgs-number_message_contatenate/2))+".png"
            #print(imageRGBPath)
            coordinates.imageSizeBefore=[480,640]
            
            imageRGB=images[num_picture]
            
            #print(f"num_msgs: {num_msgs} num_picture:{num_picture*5 }")
            """
            detector_onImage(imageRGB)
             
            """
            object_detector=Detector2(model_type="IS")
            prediction=object_detector.onImage(imageRGB )
            
            #print("prediction:" )
            #print(prediction["instances"])
            
            if len(prediction["instances"])<len(coordinates.coor):
                #print("keine objekte gefunden")
                #imageRGB=imageRGB[0:480, 0:550]
                height = 480
                width = 640
                dim = (width, height) 
                imageRGB=cv2.resize(imageRGB,dim,interpolation=cv2.INTER_AREA)
                for i in range(len(coor[0])):
                   x_min ,y_min,x_max, y_max = int (coor[0][i]),int(coor[1][i]),int(coor[2][i]),int(coor[3][i])
                   start_point=(x_min,y_min)
                   #print(f"startpoint: {start_point}")
                   end_point=(x_max,y_max)
                   color=(255,0,0)                    
                   thickness=5
                   imageRGB = cv2.rectangle(imageRGB, start_point, end_point, color, thickness)
                windowName="Bounding Box of the Events"
                cv2.imshow(windowName,imageRGB)
                cv2.waitKey(33)
                cv2.destroyAllWindows()
                
                
            end_time_identification=time.time()
            time_detectron+= (end_time_identification-start_time_identification)
            """ 
            
            imageRGB =cv2.imread(imageRGBPath)
            cv2.imshow("Result", imageRGB)
            
            key=cv2.waitKey(33)
            if key==0 :
                cv2.destroyAllWindows()
            """
            
           
            num_picture+=1
            #print(coordinates.imageSizeBefore)
            image = np.zeros((480,640))
            polarity = np.zeros(size)
            timestamp = np.zeros(size)
            count = 0;
            coordinates.coor=[]
        
       # print(num_msgs)
end_time = time.time()
print(f"Time needed to load the frames: {loading_frames_time} ")
print(f"Time needed for the object segmentation: {segmentation_time} ")
print(f"Time needed for the object identification: {time_detectron} ")
print(f"Time needed for the complet programm{end_time-start_time}")
bag.close()


print('PROCESS COMPLETE')
