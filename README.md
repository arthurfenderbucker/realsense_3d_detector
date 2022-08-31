## Realsense 3d object detector

Dependencies:
- [ROS melodic](http://wiki.ros.org/melodic/Installation/Ubuntu)
- [Anaconda](https://docs.anaconda.com/anaconda/install/linux/)

### Setup:
<sub>_tested on Ubuntu 18.04 and 20.04_</sup>

Download  the yolo model and save it to the folder _models_ e.g.
[yolov3.cfg](https://opencv-tutorial.readthedocs.io/en/latest/_downloads/10e685aad953495a95c17bfecd1649e5/yolov3.cfg)
[yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)


Environment setup
```
conda create --name py38 --file spec-file.txt python=3.8
conda activate py38
```

### Demo
```
roslaunch realsense_3d_detector rviz.launch
```
or
```
python scripts/image_publisher.py
```
