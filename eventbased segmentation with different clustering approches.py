# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 14:32:21 2023

@author: Richard
"""

import rosbag
import numpy as np
import cv2
import os

#Import for plotting: 
import matplotlib.pyplot as plt
from matplotlib import patches


#Import for the clustering
from sklearn.cluster import SpectralClustering,DBSCAN,AgglomerativeClustering, MeanShift
from sklearn.metrics import silhouette_score
 #Cleaning of the data
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import kneighbors_graph,LocalOutlierFactor
from sklearn.svm import OneClassSVM
from scipy.sparse import find,coo_matrix


ROOTDIR="D:\Masterarbeit_Experiment\extraction_scripts4"
#BAGFILE =os.path.join(ROOTDIR,"events.bag")
BAGFILE="D:/Masterarbeit_Experiment/extraction_scripts4/events.bag"
IMAGEDIR=os.path.join(ROOTDIR,"images")

#Creates from a given bag-file with events the events for each frame and returns a list the events sorted to frames 
#It will use every "step"-th frame 
def load_event_images_from_folder(bag_path,step):
    bag=rosbag.Bag(bag_path)
    images=[]
    num_messages=0
    TOPIC = '/dvs/events'
    image=np.zeros((480,640))
    
    for top,msg,tim in bag.read_messages(TOPIC):
        
        if (num_messages % step) != 0:
            num_messages+=1
            continue
                
        for event_traverser in range(len(msg.events)):
            image[(msg.events)[event_traverser].y,(msg.events)[event_traverser].x] = int((msg.events)[event_traverser].polarity)*254
      
        images.append(image)
        image=np.zeros((480,640))
        num_messages+=1
        print(len(images))
    
    return images

def load_events_from_folder(bag_path,step):
    bag=rosbag.Bag(bag_path)
    images=[]
    num_messages=0
    TOPIC = '/dvs/events'
    i =0
    
    for top,msg,tim in bag.read_messages(TOPIC):
        image=[]
        if (num_messages % step) != 0:
            num_messages+=1
            continue
                
        for event_traverser in range(len(msg.events)):
            if int((msg.events)[event_traverser].polarity) !=0:
                event=[(msg.events)[event_traverser].y,(msg.events)[event_traverser].x,int((msg.events)[event_traverser].polarity)*254]
                image.append(event)
        
        
        images.append(image)
        num_messages+=1
        print(len(images))
    return images


def cleaning_of_the_data_ISOFOREST(event_images,contamination_input):
    labels=[]
    cleaned_data=[]
    
    for i in range(len(event_images)):
        model = IsolationForest(random_state=0, n_jobs=-1, contamination=contamination_input)
        model.fit(event_images[i])
        label = model.predict(event_images[i])
        #cleaned_data = labels.reshape(data.shape)
        output=[]
        for j in range(len(label)):
            if label[j]!=-1:
                output.append(event_images[i][j])    
       
        cleaned_data.append(output)
    return cleaned_data

def cleaning_of_the_data_ONE_CLASS_SVM(event_images):
    labels=[]
    cleaned_data=[]
    
    for i in range(len(event_images)):
        model = OneClassSVM()
        model.fit(event_images[i])
        label = model.predict(event_images[i])
        #cleaned_data = labels.reshape(data.shape)
        output=[]
        for j in range(len(label)):
            if label[j]!=-1:
                output.append(event_images[i][j])    
       
        cleaned_data.append(output)
    return cleaned_data

def cleaning_of_the_data_LocalOutlierFactor(event_images,contamination_input):
    labels=[]
    cleaned_data=[]
    
    for i in range(len(event_images)):
        
        
        model = LocalOutlierFactor(n_neighbors=50,contamination=contamination_input,novelty=True)
        model.fit(event_images[i])
        label = model.predict(event_images[i])
        #cleaned_data = labels.reshape(data.shape)
        output=[]
        for j in range(len(label)):
            if label[j]!=-1:
                output.append(event_images[i][j])    
       
        cleaned_data.append(output)
    return cleaned_data


def remove_noise(sparse_matrices, threshold):
    cleaned_data = []
    for i in range(len(sparse_matrices)):
        row, col, data = find(sparse_matrices[i])
        event_counts = np.bincount(row)
        noisy_points = np.where(event_counts < threshold)[0]
        clean_sparse_matrix = coo_matrix(sparse_matrices[i])
        for point in noisy_points:
            indices = np.where(row == point)[0]
            clean_sparse_matrix.data[indices] = 0
        cleaned_data.append(clean_sparse_matrix.toarray())
    return cleaned_data



def clustering_dbscan(images,NEIGHBORS):
    corner_points=[]
    for i in range(len(images)):
        print(f"clustering_dbscan {i}")
        xy_values = np.array([d[:2] for d in images[i]])
        clustering_db = DBSCAN(eps=10, min_samples=NEIGHBORS, leaf_size=50).fit_predict(xy_values)
        
        
        corner_points_image = []
        for cluster_label in range(0,max(clustering_db)):
            cluster_indices = np.argwhere(clustering_db == cluster_label)
            #print(cluster_label)
            
                
            if cluster_indices.size > 0:
                #print(f"xy-values for the cluster {cluster_label} :{xy_values[cluster_indices]}")
                
                min_coords = np.min(xy_values[cluster_indices], axis=0)
                x_min = min_coords[0, 1]
                y_min = min_coords[0, 0]
                
                max_coords = np.max(xy_values[cluster_indices], axis=0)
                x_max = max_coords[0, 1]
                y_max = max_coords[0, 0]
                
                corner_points_cluster = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
                
                corner_points_image.append(corner_points_cluster)
        corner_points.append(corner_points_image)
    return corner_points

def clustering_meanShift(images,bandwidth_input):
    corner_points=[]
    for i in range(len(images)):
        print(f"clustering_meanShift {i}")
        xy_values = np.array([d[:2] for d in images[i]])
        clustering_db = MeanShift(cluster_all=False,bandwidth=bandwidth_input).fit_predict(xy_values)
        
        
        corner_points_image = []
        for cluster_label in range(0,max(clustering_db)):
            cluster_indices = np.argwhere(clustering_db == cluster_label)
            #print(cluster_label)
            
                
            if cluster_indices.size > 0:
                #print(f"xy-values for the cluster {cluster_label} :{xy_values[cluster_indices]}")
                
                min_coords = np.min(xy_values[cluster_indices], axis=0)
                x_min = min_coords[0, 1]
                y_min = min_coords[0, 0]
                
                max_coords = np.max(xy_values[cluster_indices], axis=0)
                x_max = max_coords[0, 1]
                y_max = max_coords[0, 0]
                
                corner_points_cluster = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
                
                corner_points_image.append(corner_points_cluster)
        corner_points.append(corner_points_image)
    return corner_points

def clustering_agglomaerative(images,distance):
    corner_points=[]
    for i in range(len(images)):
        print(f"clustering_agglomaerative {i}")
        xy_values = np.array([d[:2] for d in images[i]])
        clustering_ag =AgglomerativeClustering(distance_threshold=distance,n_clusters=None,linkage="complete").fit_predict(xy_values)
        
        
        corner_points_image = []
        for cluster_label in range(0,max(clustering_ag)):
            cluster_indices = np.argwhere(clustering_ag == cluster_label)
            
            
                
            if cluster_indices.size > 0:
                #print(f"xy-values for the cluster {cluster_label} :{xy_values[cluster_indices]}")
                
                min_coords = np.min(xy_values[cluster_indices], axis=0)
                x_min = min_coords[0, 1]
                y_min = min_coords[0, 0]
                
                max_coords = np.max(xy_values[cluster_indices], axis=0)
                x_max = max_coords[0, 1]
                y_max = max_coords[0, 0]
                
                corner_points_cluster = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
                
                corner_points_image.append(corner_points_cluster)
        corner_points.append(corner_points_image)
    
    return corner_points


def clustering_spectral_with_graph(images, max_number_clusters, NEIGHBORS):
    n_clusters_range = range(2, max_number_clusters)
    clustering_scs = []
    corner_points = []
    for i in range(len(images)):
        xy_values = np.array([d[:2] for d in images[i]])
        adMat = kneighbors_graph(xy_values, n_neighbors=NEIGHBORS)
        scores_sc = []
        max_score_sc = -20
        print(f"clustering_spectral_with_graph: {i}")
        for CLUSTERS in n_clusters_range:
            clustering_sc = SpectralClustering(n_clusters=CLUSTERS, random_state=0,
                                               affinity='precomputed_nearest_neighbors',
                                               n_neighbors=NEIGHBORS, assign_labels='kmeans',
                                               n_jobs=1).fit_predict(adMat)
            scores_sc.append(silhouette_score(xy_values, clustering_sc))
            
            if scores_sc[-1] > max_score_sc:
                max_score_sc = scores_sc[-1]
                opt_clusters_sc = CLUSTERS
        clustering_sc = SpectralClustering(n_clusters=opt_clusters_sc, random_state=0,
                                           affinity='precomputed_nearest_neighbors',
                                           n_neighbors=NEIGHBORS, assign_labels='kmeans',
                                           n_jobs=-1).fit_predict(adMat)
        clustering_scs.append(clustering_sc)
        corner_points_image = []
        for cluster_label in range(opt_clusters_sc):
            cluster_indices = np.argwhere(clustering_sc == cluster_label)
            
            
                
            if cluster_indices.size > 0:
                print(np.shape(xy_values[cluster_indices]))
                
                min_coords = np.min(xy_values[cluster_indices], axis=0)
                x_min = min_coords[0, 1]
                y_min = min_coords[0, 0]
                
                max_coords = np.max(xy_values[cluster_indices], axis=0)
                x_max = max_coords[0, 1]
                y_max = max_coords[0, 0]
                
                corner_points_cluster = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
                corner_points_image.append(corner_points_cluster)
        corner_points.append(corner_points_image)
    return clustering_scs, corner_points

def plotting (cleaned_events,events_images,corner_points_image,y,x):
    for i in range(len(events_images)):
        fig, axs = plt.subplots(y, x, figsize=(15, 6))
        axs[0,0].imshow(events_images[i], cmap='gray')
        axs[0,0].set_title('Original data')
        y_position=[d[0]for d in cleaned_events[i]]
        x_position=[d[1]for d in cleaned_events[i]]
        axs[0,1].scatter(x_position,y_position,s=1)
        axs[0,1].set_xlim(0, 640)
        axs[0,1].set_ylim(480, 0)
        axs[0,1].set_aspect('equal')
        for j in range(len(corner_points_image)):
            row = (j + 2) // x
            col = (j + 2) % x
            axs[row, col].imshow(events_images[i], cmap='gray')
            axs[row, col].set_title('Threshold={}'.format((j)*40+100))
            for k in range(len(corner_points_image[j][i])):
                rect = patches.Polygon(corner_points_image[j][i][k], linewidth=1, edgecolor='r', facecolor='none')
                axs[row, col].add_patch(rect)
    
            axs[row, col].set_xlim(0, 640)
            axs[row, col].set_ylim(480, 0)
            axs[row, col].set_aspect('equal')
    
        plt.show()
        
def plotting_for_cleaning(events_images,cleaned_events,y,x,contamination_start,contamination_step):
    for i in range(len(events_images)):
        fig,axs=plt.subplots(y,x, figsize=(15,6))
        axs[0,0].imshow(events_images[i], cmap='gray')
        axs[0,0].set_title('Original data')
        for j in range(len(cleaned_events)):
             row = (j+1) // x
             col = (j+1) % x
             axs[row, col].set_title('contamination={}'.format((j)*contamination_step+contamination_start))
             y_position=[d[0]for d in cleaned_events[j][i]]
             x_position=[d[1]for d in cleaned_events[j][i]]
             axs[row,col].scatter(x_position,y_position,s=1)
             axs[row, col].set_xlim(0, 640)
             axs[row, col].set_ylim(480, 0)
             axs[row, col].set_aspect('equal')
        plt.plot()
        
if __name__=="__main__":
    
    events_images=load_event_images_from_folder(BAGFILE,10)
    events_for_each_image=load_events_from_folder(BAGFILE,10)
    #events_images=[0,0,0,0,0,0,0,0,0]
    
    cleaned_events=cleaning_of_the_data_ISOFOREST(events_for_each_image,0.2)
    #cleaned_events=cleaning_of_the_data_LocalOutlierFactor(events_for_each_image)
    #cleaned_events=cleaning_of_the_data_LocalOutlierFactor(cleaned_events)
    #cleaned_events=cleaning_of_the_data_ISOFOREST(cleaned_events)
    #clustering_scs, corner_points=clustering_spectral_with_graph(cleaned_events,10,100)
    #cleaned_events=[]
    """
    for i in np.arange(0.05,0.5,0.05):
        cleaned_events.append(cleaning_of_the_data_ISOFOREST(events_for_each_image,i))
    """
    
    #cleaned_events=cleaning_of_the_data_ONE_CLASS_SVM(events_for_each_image) 
    
    """
    
    for i in range(len(events_images)):
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        axs[0].imshow(events_images[i], cmap='gray')
        axs[0].set_title('Original data')
        y_position=[d[0]for d in cleaned_events[i]]
        x_position=[d[1]for d in cleaned_events[i]]
        axs[1].scatter(x_position,y_position,s=1)
        axs[1].set_xlim(0, 640)
        axs[1].set_ylim(480, 0)
        axs[1].set_aspect('equal')
    """
    #corner_points=clustering_dbscan(cleaned_events,50)
    #cleaned_events=remove_noise(events_for_each_image, 10)
    #plotting_for_cleaning(events_images,cleaned_events,2,5,0.05,0.05)
    
    """Agllomaeraative Clustering for a distance from 100 to 460
    corner_points_image=[]
    for i in range(100,180,8):
        corner_points_image.append(clustering_agglomaerative(cleaned_events,i))
        
    plotting(cleaned_events, events_images,corner_points_image, 3, 4)
    """
    """MeanShift Clustering for a bandwith from 20 to 100"""
    corner_points_image=[]
    for i in range(20,100,10):
        corner_points_image.append(clustering_meanShift(cleaned_events,i))
        
    plotting(cleaned_events, events_images,corner_points_image, 2, 5)
    
        
            