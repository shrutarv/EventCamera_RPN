'''
This script will loop all rgb images, find the closest reconstructed event image and save them with the same time stamp
'''


import os
import cv2
import shutil

save_path = '/home/gouda/event_camera/ClusteringEventCamera/calibration/my_calibration_data/calib/associated_data'
rgb_path = '/home/gouda/event_camera/ClusteringEventCamera/calibration/my_calibration_data/calib/cam0'
event_path = '/home/gouda/event_camera/ClusteringEventCamera/calibration/my_calibration_data/calib/cam1'

def main():
    os.mkdir(save_path + '/cam0')
    os.mkdir(save_path + '/cam1')

    rgb_list = [int(os.path.splitext(filename)[0]) for filename in os.listdir(rgb_path)]
    rgb_first = min(rgb_list)
    rgb_list = [x - rgb_first for x in rgb_list]
    event_list = [int(int(os.path.splitext(filename)[0])/1000) for filename in os.listdir(event_path)]
    event_first = min(event_list)
    event_list = [x - event_first for x in event_list]

    for rgb_img in rgb_list:
        closest_event_reconstruct = min(event_list, key=lambda x: abs(x - rgb_img))
        rgb_img += rgb_first  # change back to record timestamp
        closest_event_reconstruct = int(closest_event_reconstruct+event_first) * 1000  # change back to record timestamp
        shutil.copy(rgb_path + '/' + str(rgb_img) + '.png', save_path + '/cam0/' + str(rgb_img) + '.png')
        shutil.copy(event_path + '/' + str(closest_event_reconstruct) + '.png', save_path + '/cam1/' + str(rgb_img) + '.png')
        #shutil.copy(event_path + '/' + str(closest_event_reconstruct) + '.png', save_path + '/cam0/' + str(rgb_img) + '_e.png')  # Debug


if __name__ == '__main__':
    main()
