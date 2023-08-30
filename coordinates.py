# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 17:08:05 2023

@author: Richard
"""
#import torch
from boxes import *

class coordinates:
    
    coor=[]
    imageSizeBefore=[]
    
    """
    
    
    coor_x_min:float
    coor_x_max:float
    coor_y_min:float
    coor_y_max:float
    """
    """koord=torch.tensor([[261.1119,245.1264,585.5193,675.5699]])
    #print (f"Demensionen{koord.dim()}")
    boundingBox=Boxes(koord)
    print(boundingBox)
    groeße=[773,1333]
    propability=torch.tensor([1.1060e+01])
    test=Instances(groeße)
    test.__setattr__("proposal_boxes", boundingBox) 
    test.__setattr__("objectness_logits", propability)
   """
        
        