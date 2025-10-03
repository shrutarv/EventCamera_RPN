# Event Camera as Region Proposal Network [[arxiv](https://arxiv.org/abs/2305.00718)]
The human eye consists of two types of photoreceptors, rods and cones. Rods are responsible for monochrome vision, and cones for color vision. The number of rods is much higher than the cones, which means that most human vision processing is done in monochrome. An event camera reports the change in pixel intensity and is analogous to rods. Event and color cameras in computer vision are like rods and cones in human vision. Humans can notice objects moving in the peripheral vision (far right and left), but we cannot classify them (think of someone passing by on your far left or far right, this can trigger your attention without knowing who they are). Thus, rods act as a region proposal network (RPN) in human vision. Therefore, an event camera can act as a region proposal network in deep learning Two-stage object detectors in deep learning, such as Mask R-CNN, consist of a backbone for feature extraction and a RPN. Currently, RPN uses the brute force method by trying out all the possible bounding boxes to detect an object. This requires much computation time to generate region proposals making two-stage detectors inconvenient for fast applications. This work replaces the RPN in Mask-RCNN of detectron2 with an event camera for generating proposals for moving objects. Thus, saving time and being computationally less expensive. The proposed approach is faster than the two-stage detectors with comparable accuracy

# Architecture

# Architecture
<p align="center">
  <img src="media/Architecture.png" width = "650" />  
</p>

## Installing the IDS camera driver.

1. Create a workspace event_ws
2. Clone the repository https://github.com/FLW-TUDO/ids_camera_driver.git
3. Install the dvxplorer driver from ultimate SLAM github repository https://github.com/uzh-rpg/rpg_ultimate_slam_open
4. Do a catkin build
5. Source the workspace
6. Execute launch file ids_dvs.launch in the launch folder

# Install detectron2
Check the repository https://github.com/shrutarv/detectron2.git to install detectron2.

