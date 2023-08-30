# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 07:08:04 2023

@author: Richard
"""
import json
oldpfad="D:\Masterarbeit_Experiment\Hall4\labels_hall4_oldRCNN_version1_GT.json"
newpfad="D:\Masterarbeit_Experiment\Hall4\labels_hall4_oldRCNN_version1_GT.json"
# Load the JSON file
with open(oldpfad, "r") as f:
    data = json.load(f)

# Modify the category id and image id
for annotation in data["annotations"]:
    if annotation["category_id"] == 1:
        annotation["category_id"] = 0
    
    annotation["image_id"]=annotation["image_id"]-1


    
for images in data["images"]:
    
    images["id"]=images["id"]-1


data["categories"][0]["id"]=0

"""
for d in data:
    d["id"]=d["id"]+1
"""    
# Save the modified JSON file
with open(newpfad, "w") as f:
    json.dump(data, f, indent=4)

