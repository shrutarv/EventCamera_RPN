# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:33:10 2023

@author: Richard
"""

from dv import NetworkEventInput
from dv import AedatFile
from datetime import *
from decord import VideoReader
from typing import List

import numpy as np
import cv2

from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
import kdtree
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import pandas as pd
from scipy.spatial import ConvexHull, convex_hull_plot_2d

def visualise(clusters,xs,ys):
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
    plt.imshow(image_filtered)
    for i in range(0,len(clusters)):
        m = i
        if m>13:
            m = 13
        plt.scatter(clusters[i][:,0], clusters[i][:,1],color = color[m],s=1)
        
        # Creating a convex hull around  objects
        hull = ConvexHull(clusters[i])
        #plt.plot(clusters[i][:,0], clusters[i][:,1], 'o')
        for simplex in hull.simplices:
            plt.plot(clusters[i][simplex, 0], clusters[i][simplex, 1], color = 'pink')
        
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
            
    return x_mi, x_ma, y_mi, y_ma, bb_areas

def clustering(data,e, ms):
    clusters = []
    # eps: The distance that specifies the neighborhoods. Two points are considered to be neighbors if the distance between them are less than or equal to eps.
    # min_samples(ms): Minimum number of data points to define a cluster.
    try:
        
        clustering =DBSCAN(eps=e,min_samples=ms,algorithm='kd_tree',  n_jobs = 6,).fit(data)
        
        # list of every cluster, including noise (assigned to -1).
        found_clusters = set(clustering.labels_)
        #print(found_clusters)
        # remove noise
        try:
            found_clusters.remove(-1)
        except Exception as e:
            print("No noise")
        
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

def boundingBox():
    bb_areas = np.zeros(len(clusters))
    x_min, x_max, y_min, y_max, bb_areas = min_max(clusters, bb_areas)
    
    coord = [[x_min,y_min], [x_min,y_max], [x_max,y_max], [x_max,y_min]]
   
    coord.append(coord[0])
    xs, ys = zip(*coord) #create lists of x and y values
    visualise (clusters,xs,ys)
    print(xs,ys)
    
def with_decord(video: str) -> List[int]:
    """
    Link: https://github.com/dmlc/decord
    My comments:
        Works really well, but it seems to only work with mkv and mp4 file.
        Mp4 file can have a +- 1 ms difference with ffms2, but it is acceptable.

    Parameters:
        video (str): Video path
    Returns:
        List of timestamps in ms
    """
    vr = VideoReader(video)

    timestamps = vr.get_frame_timestamp(range(len(vr)))
    #timestamps = (timestamps[:, 0]*1000).round().astype(float).tolist()/1000
    timestamps=timestamps[:,1]-timestamps[:,0]
    for i in range(1,len(timestamps)):
        timestamps[i]=timestamps[i]+timestamps[i-1]
    return timestamps






if __name__=="__main__" :
    video=r"C:/Users/Richard/spyder-py3/geeksforgeeks/Stockumer.mp4"
    decord_timestamps = with_decord(video)
    image = np.zeros((480,640))
    eps = 30
    ms = 50
    time =80000
    x = np.zeros(time)
    y = np.zeros(time)
    p=0
    frame=1
    kernel = np.ones((2, 1), np.uint8)
    print(decord_timestamps)
    with AedatFile('D:/Masterarbeit/videosexamples/output1.aedat4') as i:
        for event in i['events']:
            if p==0:
                startEventTime=event.timestamp
                p+=1
            x[p] = event.x
            y[p] = event.y
            image[event.y,event.x] = int(event.polarity)*254
            
            deltaEvent=(event.timestamp-startEventTime)/1000000
            
            
            
            if deltaEvent>decord_timestamps[frame]:
                print(f"aktualles Event {deltaEvent}       timestamp frame {decord_timestamps[frame]}")
                
                image_filtered = cv2.erode(image, kernel, iterations=1)
                
                yx_coords = np.column_stack(np.where(image_filtered > 0))
                y_temp = yx_coords[:,0]
                x_temp = yx_coords[:,1]
                x_t = x_temp.reshape(len(x_temp),1)
                y_t = y_temp.reshape(len(y_temp),1)
                data = np.concatenate((x_t,y_t),1)
                #print(data)
                clusters = clustering(data, eps, ms)
                buffer = 5.0
                threshold = 100 
                boundingBox()
                image[:,:] = 0
                frame+=1
                