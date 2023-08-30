# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 12:48:37 2023

@author: Richard
"""

class stoppingtime:
    start_time=[]
    end_time=[]
    
    def needed_time():
               
        for i in range(len(stoppingtime.start_time)):
            print(str(stoppingtime.end_time[i]-stoppingtime.start_time[i]).replace(".",","))
    
    
    