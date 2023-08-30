import cv2
import numpy as np


def make_intrinsic(fx, fy, cx, cy):
    return np.array([
        [fx, 0,  cx ],
        [0,  fy, cy],
        [0,  0,  1]
    ])

rgb_img = cv2.imread('/home/gouda/event_camera/ClusteringEventCamera/calibration/my_calibration_data/calib/rect/rgb.png')
event_img = cv2.imread('/home/gouda/event_camera/ClusteringEventCamera/calibration/my_calibration_data/calib/rect/event.png')

k_left = make_intrinsic(641.5243678901976, 641.7464737763148, 353.52477924858874, 222.1000677349316)
d_left= np.array([0.051175544074370506, 0.23024244620721399, -0.015952671187329157, 0.010578967975497495,0])
k_right = make_intrinsic(680.1079435529213, 689.599770982309, 371.7706217885453, 254.60011716683675)
d_right = np.array([-0.451261798544749, 1.4142362363107792, 0.0034429925845535806, 0.01262572569842129,0])
c_ext = np.array([
    [0.994737471254337, -0.0038054519290324374, -0.10238594541317993, 0.06967449650859936],
    [0.0017533845240814366, 0.999795934005254, -0.02012500905043728, -0.010319773249440733],
    [0.10244163667789356, 0.019839578679631632, 0.99454115158327, 0.017250476165564705],
    [0.0, 0.0, 0.0, 1.0]
])

R = c_ext[0:3,0:3]
T = c_ext[0:3, 3]
img_height = 480
img_width = 640

r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(k_left, d_left,
                                                  k_right, d_right,
                                                  (img_height, img_width),
                                                  R,T,flags=cv2.CALIB_ZERO_DISPARITY)

map1x, map1y = cv2.initUndistortRectifyMap(
    cameraMatrix=k_left,
    distCoeffs=d_left,
    R=r1,
    newCameraMatrix=p1,
    size=(img_width, img_height),
    m1type=cv2.CV_32FC1)

map2x, map2y = cv2.initUndistortRectifyMap(
    cameraMatrix=k_right,
    distCoeffs=d_right,
    R=r2,
    newCameraMatrix=p2,
    size=(img_width, img_height),
    m1type=cv2.CV_32FC1)
img1_rect = cv2.remap(rgb_img, map1x, map1y, cv2.INTER_LINEAR,cv2.BORDER_CONSTANT)
img2_rect = cv2.remap(event_img, map2x, map2y, cv2.INTER_LINEAR,cv2.BORDER_CONSTANT)

cv2.imwrite("/home/gouda/event_camera/ClusteringEventCamera/calibration/my_calibration_data/calib/rect/rect_rgb.png", img1_rect)
cv2.imwrite("/home/gouda/event_camera/ClusteringEventCamera/calibration/my_calibration_data/calib/rect/rect_event.png", img2_rect)

cv2.imshow("rgb", img1_rect)
cv2.waitKey()
cv2.imshow("event", img2_rect)
cv2.waitKey()
cv2.destroyAllWindows()

