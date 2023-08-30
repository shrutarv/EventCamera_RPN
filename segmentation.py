# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:59:59 2022

@author: shrut
"""
import pandas as pd
from dv import NetworkEventInput
import numpy as np
import cv2
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
import kdtree
from sklearn.cluster import DBSCAN
import math
from Node_With_Payload import *
from Bot import *
from ObjData import *
from dv import AedatFile
from ConcaveHull import ConcaveHull
from shapely.geometry import asPolygon
from shapely.geometry import asLineString
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from shapely.geometry import Polygon
import timeit
#import datetime as dt
import time

def intersection():
        
    p1 = Polygon([(0,0), (1,1), (1,0)])
    p2 = Polygon([(0,1), (1,0), (1,1)])
    print(p1.intersects(p2))

def getCentroids(clusters):
    centroids = []
    for cluster in clusters:
        meanY = int(np.mean(cluster[:,0]))
        meanX = int(np.mean(cluster[:,1]))
        centroids.append([meanY, meanX])
    return centroids

def getAngleFromTwoCentroids(c1,c2):
    angle = np.arctan2((c2[0] - c1[0]), (c2[1]-c1[1]))
    angle = math.degrees(angle)
    return angle

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

def boundingBox():
    bb_areas = np.zeros(len(clusters))
    x_min, x_max, y_min, y_max, bb_areas = min_max(clusters, bb_areas)
    
    coord = [[x_min,y_min], [x_min,y_max], [x_max,y_max], [x_max,y_min]]
    coord.append(coord[0])
    xs, ys = zip(*coord) #create lists of x and y values
    # uncomment to visualize
    #visualise (clusters,xs,ys)
   
    
def kmeans(x_t,y_t,data):
    kmeans = KMeans(3, max_iter=500, random_state=4)
    kmeans.fit(data)
    identified_clusters = kmeans.fit_predict(data)
        
    #data_with_clusters['Clusters'] = identified_clusters 
    plt.scatter(x_t,y_t,identified_clusters,cmap='rainbow')

def calculate_distance(centroid):
    distance = np.zeros((250,3))
    j = 0
    tree = KD_Tree()
    for neighbours in centroid:
        # The centroid of each cluster is analysed sequentially
        centroid_x = neighbours[0]
        centroid_y = neighbours[1]
        if not tree.tree:
            tree.addNode(centroid_x, centroid_y)
        else:
            # find distance to nearest neighbor
            neighbor = tree.NN(centroid_x, centroid_y)
            distanceFromNeighbor = neighbor[1]
            node= neighbor[0]
            # distance matriy has the coords of centroid of the nearest 
            # cluster and the distance of the centroid to current node centroid
            distance[j,0] = np.array(node.data.coords)[0]
            distance[j,1] = np.array(node.data.coords)[1]
            distance[j,2] = distanceFromNeighbor
            j = j + 1
            # if distance is too big, new neighbour
            if distanceFromNeighbor > threshold:
                tree.addNode(centroid_x, centroid_y)
            
            # if a prev node was at this location
            else:
                tree.updateNode(centroid_x, centroid_y, neighbor)
        return tree, distance
    
def tracker(frames):
    
    drawn_frames = []
    
    # Init a tree for tracking
    tree = KD_Tree()

    # The maximum euclidian distance at which a found neighbor in future frames can be considered to be the same bot at a new position.
    bot_width = 100

    # Stepping through each frame...
    for frame in frames:

        # Detect all the clusters in the frame (which do not belong to noise)
        clusters = clustering(frame, 15 , 170)

        for cluster in clusters:

            # Find the center point of each detected cluster
            centroid = get_centroid(cluster)

            bot_x = centroid[0]
            bot_y = centroid[1]

            # First bot detected case
            if not tree.tree: # will be true if tree is empty
                tree.addNode(bot_x, bot_y)

            else:
                # Find the distance between the current detection and the nearest neighbor
                neighbor = tree.NN(bot_x, bot_y)
                distanceFromNeighbor = neighbor[1]

                # If the distance is too great, this is a new bot
                if distanceFromNeighbor > bot_width:
                    tree.addNode(bot_x, bot_y)

                # Otherwise, a previous bot was at this location and the info on that older bot should be updated
                else:
                    tree.updateNode(bot_x, bot_y, neighbor)
        
        drawn_frames.append(frame, tree.asList())

    return drawn_frames, tree

print('PROCESS COMPLETE')

            
if __name__ == "__main__" :     
    image = np.zeros((480,640))
    time_window = 800000
    # eps: The distance that specifies the neighborhoods. Two points are considered to be neighbors if the distance between them are less than or equal to eps.
    # min_samples(ms): Minimum number of data points to define a cluster.
    # ws : this parameters is defined to accumulate the previous ws events
    ws = 20000
    eps = 30
    ms = 50
    x = np.zeros(time_window)
    y = np.zeros(time_window)
    p = 0
    t = 0
    time_counter = 1
    kernel = np.ones((2, 1), np.uint8)
    pre_recorded = True
    start = time.time()
    flag = True
    
    with AedatFile('C:/Users/shrut/output1.aedat4') as i:
    # with NetworkEventInput(address='127.0.0.1', port=64443) as i:
        for event in i['events']:
            if(flag):
                start_t = event.timestamp
                flag = False
            print(event.timestamp)
        #for event in i:
            #print(event.timestamp)
            x[p] = event.x
            y[p] = event.y
            image[event.y,event.x] = int(event.polarity)*254
            #image[int(y[p-25000]), int(x[p-25000])] = 0
            #x_temp = x[p-25000:p]
            #y_temp = y[p-25000:p]
            p += 1
            #print(p)
            #if p == time_window:
            if(time.time()-start>time_counter):
            #if p % ws ==0:
                # Perform ersosion to remove noisy events
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
                clusters = clustering(data, eps, ms)
                buffer = 20.0
                threshold = 100  
                # Calculating bounding box coordinates
                #x_min, x_max, y_min, y_max = min_max(clusters)
                boundingBox()
                #coord = [[x_min,y_min], [x_min,y_max], [x_max,y_max], [x_max,y_min]]
                #coord.append(coord[0])
                #xs, ys = zip(*coord) #create lists of x and y values
                #visualise (clusters,xs,ys)
                image[:,:] = 0    
                # Create concave hull of the clusters
                p = 0
                time_counter += 1              
                #centroid = getCentroids(clusters)
                ##hull = ConvexHull(clusters[0])
                #plt.plot(clusters[0][:,0], clusters[0][:,1], 'o')
                #for simplex in hull.simplices:
                 #   plt.plot(clusters[0][simplex, 0], clusters[0][simplex, 1], 'k-')
                #p = 0
                #tree, distance = calculate_distance(centroid)
                # Calculating distance between the centroid of each cluster
                
       # Init an empty kd-tree
              #  drawn_frames, tree = tracker(data)                        
                        

   