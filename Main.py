
from PIL import Image as ig
import matplotlib.pyplot as plt
import seaborn as sns
import torch
sns.set()
from sklearn.cluster import DBSCAN
# import datetime as dt
import time

import os
import threading
import rospy
import rosbag

from sensor_msgs.msg import Image
from dvs_msgs.msg import EventArray

from cv_bridge import CvBridge

from Detector2 import *

import numpy as np
import pycocotools.mask as maskUtils

# ROOT_DIR = 'D:/Masterarbeit_Experiment/Hall4'

# BAGFILE = "D:/Masterarbeit_Experiment/extraction_scripts2/events.bag"
BAGFILE = "/home/eventcamera/event_camera/ClusteringEventCamera/Recording/events.bag"
predictions_json1 = "/home/eventcamera/event_camera/ClusteringEventCamera/Recording/output.json"
image_path = "/home/eventcamera/event_camera/ClusteringEventCamera/Recording/_cam0_1280/"
object_detector = Detector2(model_type="IS")
if torch.cuda.is_available():
    device = torch.device("cuda")
# new_image_available = threading.Event()
TOPIC = '/dvs/events'
rgb_topic = '/cam1/image_raw'
event_topic = '/dvs/events'
size = 400000
x = np.zeros(size)
y = np.zeros(size)
image = np.zeros((480, 640))
polarity = np.zeros(size)
timestamp = np.zeros(size)
count = 0
num_msgs = 0
segmentation_time = 0
time_detectron = 0
ws = 20000
rgb_callback_time = time.time()
current_time = time.time()

flag = 1

start_loading_frames = time.time()
# images=load_images_from_folder("D:/Masterarbeit_Experiment/extraction_scripts3/images/",number_message_contatenate)
end_loading_frames = time.time()
loading_frames_time = end_loading_frames - start_loading_frames

# predictions_json2="D:/Masterarbeit_Experiment/extraction_scripts3/extraction_scripts3_output2.json"
num_picture = 0
abbruch = False
predicted_annotations1 = []
predicted_annotations2 = []
predicted_annotation2_images = []
predicted_annotation2_annotations = []
image_id = 0
detection_id = 0

event_image = np.zeros((480, 640))


def visualise(clusters, xs, ys, image_f):
    global rgb_callback_time
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
    # plt.imshow(image)
    plt.imshow(image_f)
    for i in range(0, len(clusters)):
        m = i
        if m > 13:
            m = 13
        plt.scatter(clusters[i][:, 0], clusters[i][:, 1], color=color[m], s=1)
        # plt.scatter(clusters[1][:,0], clusters[1][:,1],color = 'cyan',s=1)
        # plt.scatter(clusters[4][:,0], clusters[4][:,1],color = 'red', s=1)
        # plt.scatter(clusters[3][:,0], clusters[3][:,1],color = 'blue', s=1)
        # plt.scatter(clusters[2][:,0], clusters[2][:,1],color = 'green', s=1)
        # plt.scatter(clusters[6][:,0], clusters[6][:,1],color = 'green', s=1)
        # plt.scatter(clusters[5][:,0], clusters[5][:,1],color = 'brown', s=1)
        # plt.scatter(clusters[8][:,0], clusters[8][:,1],color = 'pink', s=1)
        # plt.scatter(clusters[9][:,0], clusters[9][:,1],color = 'pink', s=1)

        # Creating a convex hull around  objects
        # hull = ConvexHull(clusters[i])
        # plt.plot(clusters[i][:,0], clusters[i][:,1], 'o')
        # for simplex in hull.simplices:
        #   plt.plot(clusters[i][simplex, 0], clusters[i][simplex, 1], color = 'pink')

    plt.plot(xs, ys)
    plt.savefig('/home/eventcamera/event_camera/ClusteringEventCamera/Recording/bb/' + str(rgb_callback_time) + '.png')
    # print(xs,ys)
    # plt.show()


def min_max(clusters, bb_areas):
    x_mi = np.zeros(len(clusters))
    x_ma = np.zeros(len(clusters))
    y_mi = np.zeros(len(clusters))
    y_ma = np.zeros(len(clusters))
    m = 0
    for i in range(0, len(clusters)):
        x_min = np.min(clusters[i][:, 0])
        x_max = np.max(clusters[i][:, 0])
        y_min = np.min(clusters[i][:, 1])
        y_max = np.max(clusters[i][:, 1])
        area = (x_max - x_min) * (y_max - y_min)
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


def clustering(data, e, ms):
    clusters = []
    # eps: The distance that specifies the neighborhoods. Two points are considered to be neighbors if the distance between them are less than or equal to eps.
    # min_samples(ms): Minimum number of data points to define a cluster.
    try:
        clustering = DBSCAN(
            eps=e,
            min_samples=ms,
            algorithm='kd_tree',
            n_jobs=6,
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

        # visualise(clusters)
    except Exception as e:
        # we allow this to fail gracefully if no detections are found
        print('Error during clustering:{}'.format(e))
    return clusters


def boundingBox(cluster, image_filtered):
    bb_areas = np.zeros(len(cluster))
    x_min, x_max, y_min, y_max, bb_areas = min_max(cluster, bb_areas)

    coord = [[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]]
    coord.append(coord[0])
    xs, ys = zip(*coord)  # create lists of x and y values
    # uncomment to visualize
    # visualise(cluster,xs,ys,image_filtered)
    # print(f"x_min:{x_min}  y_min:{y_min}      x_max:{x_max}  y_max:{y_max}")
    return x_min * 1.1, x_max * 1.2, y_min, y_max * 1.1


def segment(image):
    kernel = np.ones((2, 1), np.uint8)

    image_filtered = cv2.erode(image, kernel, iterations=1)
    image_new = ig.fromarray(image_filtered, mode='L')
    # cv2.imshow("filtered image", image_filtered)
    yx_coords = np.column_stack(np.where(image_filtered > 0))
    y_temp = yx_coords[:, 0]
    x_temp = yx_coords[:, 1]
    # plt.figure()
    # image_new.save('/home/eventcamera/event_camera/ClusteringEventCamera/Recording/filtered/' + str(rgb_callback_time) + '.png')

    # x_temp = x[p-ws:p]
    # y_temp = y[p-ws:p]
    # x_t = x_temp.reshape(ws,1)
    # y_t = y_temp.reshape(ws,1)
    x_t = x_temp.reshape(len(x_temp), 1)
    y_t = y_temp.reshape(len(y_temp), 1)
    data = np.concatenate((x_t, y_t), 1)
    eps = 30
    ms = 50

    clusters = clustering(data, eps, ms)
    # Calculating bounding box coordinates
    # x_min, x_max, y_min, y_max = min_max(clusters)
    x_min, x_max, y_min, y_max = boundingBox(clusters, image_filtered)
    # coord = [[x_min,y_min], [x_min,y_max], [x_max,y_max], [x_max,y_min]]
    # coord.append(coord[0])
    # xs, ys = zip(*coord) #create lists of x and y values
    # visualise (clusters,xs,ys)
    image[:, :] = 0
    return [x_min, y_min, x_max, y_max]


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


def load_images_from_folder(path, step):
    images = []
    lst = os.listdir(path)
    for i in range(0, len(lst), step):
        print(i)
        img = cv2.imread((path + str(int(i)) + ".png"))

        if img is not None:
            images.append(img)
    return images


import numpy as np
from skimage.measure import find_contours


def bitmap_to_polygon(bitmap):
    """Convert a binary mask represented as a 2D numpy array into a polygon representation."""

    bitmap = np.array(bitmap)
    contours = find_contours(bitmap, 0.5)
    polygon = []
    for contour in contours:
        contour = np.fliplr(contour) - 0.5
        polygon.append(contour.tolist())
    return polygon


def bitmap_to_rle(bitmap):
    """Convert a binary mask represented as a 2D list into RLE representation."""
    mask = np.array(bitmap, dtype=np.uint8)
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return runs.tolist()


# Do it in real time. As ps of rgb camera is much lower. Therefore, whenever we get an rgb image run the segmentation


def event_callback(event_data):
    global event_image
    event_image = np.zeros((480, 640))
    for event in event_data.events:
        x = event.x
        y = event.y
        polarity = event.polarity

        # Update the accumulated image based on the event
        event_image[y, x] += polarity * 255  # Accumulate events by adjusting pixel intensity


# TODO: add a global timer. in the callback check if some amount of time has passed since the previous callback.
# If the specified amount of time has not passed then dont call the execute script, else call it.
def rgb_callback(rgb_data):
    global flag
    global rgb_callback_time
    global current_time
    if flag == 1:

        cv_image = bridge.imgmsg_to_cv2(rgb_data, desired_encoding='passthrough')
        x1, y1 = 240, 44  # Top-left corner
        x2, y2 = 1875, 1240  # Bottom-right corner

        # Crop the image
        cropped_image = cv_image[y1:y2, x1:x2]
        cropped_image = cv2.resize(cropped_image, (640, 480))
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('/home/eventcamera/event_camera/ClusteringEventCamera/Recording/rgb_image/' + str(rgb_callback_time) + '.png', cropped_image)
        cv_image_3d = cropped_image[:, :, :3]
        execute_script(cv_image_3d)
        # new_image_available.set()
        rgb_callback_time = time.time()
        flag = 0
    else:
        current_time = time.time()
        if (current_time - rgb_callback_time) > 0.5:
            # print('pass')
            flag = 1

    # cv2.imshow('RGB image',cv_image_3d)
    # rgb_timestamp = int(rgb_data.header.stamp.nsecs / 1000) + rgb_data.header.stamp.secs * 1000000
    # rgb_timestamp *= 1000
    # cv2.imwrite(cam0_dir + '/' + str(rgb_timestamp) + '.png', cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    # imageRGB = cv2.imread(cam0_dir + '/' + str(rgb_timestamp) + '.png')
    # cv2.imwrite(cam0_dir + '/' + str(i) + '.png', cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

    # bag = rosbag.Bag(save_path + '/events.bag', 'w')

    # cam0_dir = save_path + '/_cam0_1280'
    # os.mkdir(cam0_dir)

    # print("Finished Collection")


'''
def main_thread_function():
    global new_image_available

    root = tk.Tk()  # Create the Tkinter root window
    label = Label(root, text="Tkinter Label")
    label.pack()
    coor = segment(x, y, event_image)
    while not rospy.is_shutdown():
        if new_image_available.wait(timeout=0.1):  # Wait for the event with timeout
            new_image_available.clear()  # Reset the event
            # Update the Tkinter label or perform other GUI operations
            label.config(text="New Image Received")

        root.update()  # Update the Tkinter window

    root.destroy()
'''


def execute_script(cv_image_3d):
    # global event_image
    start_t = time.time()
    coor = segment(event_image)
    end_t = time.time()
    print("time for clustering and bounding box creation" + str(end_t - start_t))

    for i in range(len(coor[0])):
        coordinates.coor.append([coor[0][i], coor[1][i], coor[2][i], coor[3][i]])
    # event_image = np.zeros((480, 640))
    print(coor)

    # x = np.zeros(size)
    # y = np.zeros(size)

    coordinates.imageSizeBefore = [480, 640]

    # imageRGB=images[num_picture]
    # print(num_msgs)

    """
    detector_onImage(imageRGB)

    """
    start_time = time.time()
    prediction = object_detector.onImage(cv_image_3d)
    end_detectron_time = time.time()

    coordinates.coor = []

    """        
    predicted_annotations2={"images":predicted_annotation2_images,"annotations":predicted_annotation2_annotations,"categories":{"id":0,"name":"Person"}}
    categories={"categories":{"id":0,"name":"Person"}}
    #predicted_annotations1.append(categories)      
    """

    # print(num_msgs)
    end_time = time.time()
    print(f"Time needed for the detectron{end_time - end_detectron_time}")
    print(f"Time needed after the detectron{end_time - start_time}")

    # with open(predictions_json1, 'w') as f:
    #   json.dump(predicted_annotations1, f)
    """
    with open(predictions_json2, 'w') as f:
        json.dump(predicted_annotations2 , f)
    """
    # print('PROCESS COMPLETE')


if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)

    rgb_sub = rospy.Subscriber(rgb_topic, Image, rgb_callback)
    event_sub = rospy.Subscriber(event_topic, EventArray, event_callback)
    bridge = CvBridge()
    rospy.spin()

    event_sub.unregister()
    rgb_sub.unregister()

    '''
    bag = rosbag.Bag(BAGFILE)
    total_msgs = bag.get_message_count()
    total_img = np.size(os.listdir(image_path))
    step = total_msgs/total_img
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
    number_message_concatenate = 5
    
    start_loading_frames=time.time()
    #images=load_images_from_folder("D:/Masterarbeit_Experiment/extraction_scripts3/images/",number_message_contatenate)
    end_loading_frames=time.time()
    loading_frames_time=end_loading_frames-start_loading_frames
    
    
    #predictions_json2="D:/Masterarbeit_Experiment/extraction_scripts3/extraction_scripts3_output2.json"
    num_picture=0
    abbruch=False
    predicted_annotations1=[]
    predicted_annotations2=[]
    predicted_annotation2_images=[]
    predicted_annotation2_annotations=[]
    image_id=0
    detection_id=0
    group_value = 0
    rgb_counter = 0
    group_value = group_value + step
    for top,msg,tim in bag.read_messages(TOPIC):
        start_time_segmentation=time.time()
        #print(msg.header)
        #print("next message")

        num_msgs += 1 

        
        #for message_traverser in range(num_msgs):
        for event_traverser in range(len(msg.events)):
            #timestamp.append((msg.events)[event_traverser].ts.secs)
            
            if (num_msgs%5 !=0):
            #if(num_msgs%2 == 0 or num_msgs%5 == 0 ): # 
                break
            
            x[count] = (msg.events)[event_traverser].x
            y[count] = (msg.events)[event_traverser].y
            polarity = (msg.events)[event_traverser].polarity
            image[(msg.events)[event_traverser].y,(msg.events)[event_traverser].x] = int((msg.events)[event_traverser].polarity)*254
            count += 1
            #print(event_traverser)
        if num_msgs%int(group_value) == 0 :
            group_value = group_value + step
            coor=segment(x,y, image)
           
            end_time_segmentation=time.time()
            segmentation_time+= (end_time_segmentation-start_time_segmentation)
            print(coor)
            #print(coor)      
            print("execute segmentation script")
            start_time_identification=time.time()
            for i in range(len(coor[0])):
                coordinates.coor.append([coor[0][i],coor[1][i],coor[2][i],coor[3][i]])
            
            #print(coordinates.coor)
            x = np.zeros(size)
            y = np.zeros(size)
            rgb_counter = rgb_counter + 1
            imageRGBPath = image_path + str(rgb_counter) + ".png"
            imageRGB = cv2.imread(imageRGBPath)
            print(imageRGBPath)
            coordinates.imageSizeBefore=[480,640]
            
            #imageRGB=images[num_picture]
            #print(num_msgs)
            
            """
            detector_onImage(imageRGB)
             
            """
            object_detector=Detector2(model_type="IS")
            prediction=object_detector.onImage(imageRGB)
            print(len(prediction["instances"]))
            
            for i in range(len(prediction["instances"])): 
                print (f"durchläufe {i}")
                file_name=str(num_msgs-number_message_contatenate)+".png"
                print(file_name)
                if i>0:
                    print("Hallejuja")
                    abbruch=True
                    
                """   
                prediction_mask=prediction["instances"].__getattr__("pred_masks")[0].tolist()
                #prediction_mask=bitmap_to_rle(prediction_mask)
                #print(f"prediction_mask {prediction_mask}")
                #for pred in prediction:
                #    predicted_annotations.append(pred)
                
                prediction_mask = reduce(lambda z, y :z + y, prediction_mask)
                """


                mask = prediction["instances"].__getattr__("pred_masks")[0].tolist()
                mask = np.array(mask, order="F")
                prediction_mask = maskUtils.encode(mask)
                prediction_mask_counts=prediction_mask["counts"].decode()
                #prediction_mask = {"counts": rle["counts"], "size": [mask.shape[0], mask.shape[1]]}

                
                image_height=480
                image_width=640
               #size=[image_height,image_width]
                bBox=prediction["instances"].__getattr__("pred_boxes")
                pred_class=prediction["instances"].__getattr__("pred_classes").tolist()[i]
                score=prediction["instances"].__getattr__("scores").tolist()[i]
                bBox=bBox.tensor.tolist()
                bBox=bBox[i]
                bBox=[bBox[0],bBox[1],abs(bBox[2]-bBox[0]),abs(bBox[3]-bBox[1])]
                print(len(bBox))
                print(bBox)
                predicted_annotation1={"image_id":image_id-2,"file_name": file_name,"id":detection_id,"category_id":pred_class, "bbox":bBox,
                    "score":score,                  
                    "segmentation":{
                        "size":prediction_mask["size"],
                        "counts":prediction_mask_counts}
                    }
                #"bbox":{"0":bBox[i][0],"1":bBox[i][1],"2":bBox[i][2],"3":bBox[i][3]}
                
                
                predicted_annotations1.append(predicted_annotation1)
                """
                predicted_annotation2_image={"id":image_id,"image_width":image_width,"image_height":image_height,"file_name:": file_name}
                predicted_annotation2_annotation={"id":image_id,"category_id":pred_class,"segmentation":{i:prediction_mask},"bbox":bBox}
                predicted_annotation2_images.append(predicted_annotation2_image)
                predicted_annotation2_annotations.append(predicted_annotation2_annotation)
                """
                detection_id+=1
            image_id+=1    
            
            
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
    """        
    predicted_annotations2={"images":predicted_annotation2_images,"annotations":predicted_annotation2_annotations,"categories":{"id":0,"name":"Person"}}
    categories={"categories":{"id":0,"name":"Person"}}
    #predicted_annotations1.append(categories)      
    """        
    
        
       # print(num_msgs)
end_time = time.time()
print(f"Time needed to load the frames: {loading_frames_time} ")
print(f"Time needed for the object segmentation: {segmentation_time} ")
print(f"Time needed for the object identification: {time_detectron} ")
print(f"Time needed for the complet programm{end_time-start_time}")
bag.close()


#json.dumps(predicted_annotations )
#predicted_annotations.append(prediction)
with open(predictions_json1, 'w') as f:
    json.dump(predicted_annotations1 , f)
"""
with open(predictions_json2, 'w') as f:
    json.dump(predicted_annotations2 , f)
"""
print('PROCESS COMPLETE')
'''
