from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

import numpy as np
import skimage.io as io

from pycocotools import mask as maskUtils
import cv2

def evaluate_iou(gt_json, results_json, img_id, annType):
    cocoGt = COCO(gt_json)
    cocoDt = cocoGt.loadRes(results_json)
    
    # set evaluation parameters
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = [img_id]
    
    # evaluate results
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    # get predicted and ground truth annotations for the image
    annIds = cocoGt.getAnnIds(imgIds=img_id)
    anns = cocoGt.loadAnns(annIds)
    dt_annIds = cocoDt.getAnnIds(imgIds=img_id)
    dt_anns = cocoDt.loadAnns(dt_annIds)
    
    # load the image
    img_info = cocoGt.loadImgs(img_id)[0]
    img_path = "{}/{}".format('D:/Masterarbeit_Experiment/Hall1/images/', img_info['file_name'])
    img = io.imread(img_path)

    # draw predicted bounding boxes in green
    for dt_ann in dt_anns:
        bbox = dt_ann['bbox']
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int( bbox[3])), (0, 255, 0), 2)

    # draw ground truth bounding boxes in red
    for ann in anns:
        bbox = ann['bbox']
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (255, 0, 0), 2)

    # show the image with bounding boxes
    io.imshow(img)
    io.show()
    
if __name__=="__main__":

    gt_json = "D:\Masterarbeit_Experiment\outside3\labels_outside3version2_GT.json"
    
    # Results from different methods
    # detectron2 trained on DoPose
    #results_json = '/run/user/1002/gvfs/smb-share:server=129.217.152.32,share=data/Anas_Gouda/unseen_object_classification/train_dopose_detectron/train_1/inference/coco_instances_results.json'
    # GT masks + DoClassify
    results_json = "D:\Masterarbeit_Experiment\outside3\outside3_output2.json"
    
    annType = 'bbox'
    #annType = 'segm'
    
    
       
    
    cocoGt=COCO(gt_json)
    print("schritt0")
    cocoDt=cocoGt.loadRes(results_json)
    print("schritt1")
    imgIds=sorted(cocoGt.getImgIds())
    """
    # Loop through images and get annotations
    for imgId in imgIds:
        # Get ground truth annotations
        gt_anns = cocoGt.loadAnns(cocoGt.getAnnIds(imgId))
        # Get predicted annotations
        dt_anns = cocoDt.loadAnns(cocoDt.getAnnIds(imgId))
    
        # Get ground truth bounding boxes
        gt_boxes = [ann['bbox'] for ann in gt_anns]
        print(f"Ground truth boxes for image {imgId}: {gt_boxes}")
    
        # Get predicted bounding boxes
        dt_boxes = [ann['bbox'] for ann in dt_anns]
        print(f"Predicted boxes for image {imgId}: {dt_boxes}")
    
    # Run COCO evaluation as before
    """
    print("schritt2")
    # running evaluation
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    print("schritt3")
    cocoEval.params.imgIds = imgIds
    print("schritt4")
    cocoEval.evaluate()
    print("schritt5")
    cocoEval.accumulate()
    print("schritt6")
    cocoEval.summarize()
    """
    for i in range(1,23,1):
            evaluate_iou(gt_json, results_json, i, annType)
       """     