# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 12:10:05 2023

@author: Richard
"""
from Detector2 import *
import cv2

pfad="D:/Masterarbeit_Experiment/outside3/images/40.png"

image=cv2.imread(pfad)

object_detector=Detector2(model_type="IS")
#object_detector.onImage("D:/Masterarbeit/Test/bild305.png")
#object_detector.onImage("C:/Users/Richard/spyder-py3/geeksforgeeks/ebertplatz.jpg")


object_detector.onImage(image)
#object_detector.onVideo("C:/Users/Richard/spyder-py3/geeksforgeeks/Stockumer.mp4")


