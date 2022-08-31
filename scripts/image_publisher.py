#!/usr/bin/env python

import pyrealsense2 as rs
import cv2
import numpy as np
import time
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--show', type=int, default=0)
parser.add_argument('--ros', type=bool, default=False)
args, unknown = parser.parse_known_args()

ros = args.ros
show = True if args.show == 1 else False


model_path="/home/arthur/Downloads/"


if ros:
    import rospy
    import roslib
    from sensor_msgs.msg import Image
    import json
    import rospkg
    import os
    from visualization_utils import *
    
    pkg_path=rospkg.RosPack().get_path('realsense_3d_detector')
    conf_file_path = os.path.join(pkg_path, 'config', 'obj_stl.json')

    model_path = os.path.join(pkg_path, "models/")
    
    obj_stl = {}
    with open(conf_file_path) as f:
        obj_stl = json.load(f)

    rospy.init_node('image_publisher', anonymous=True)
    image_pub = rospy.Publisher("/camera/image_raw/",Image,queue_size=1)

    marker_pub = rospy.Publisher("/objs_markers", MarkerArray,queue_size=1)

    def cv2_to_imgmsg(cv_image):
        img_msg = Image()
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        img_msg.encoding = "bgr8"
        img_msg.is_bigendian = 0
        img_msg.data = cv_image.tostring()
        img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
        return img_msg

    def pub_markers(obj_names, obj_poses, colors=None):

        markerArray = MarkerArray()
        markerArray.markers.append(clean_marker())

        for i in range(len(obj_names)):
           
            if obj_names[i] in obj_stl.keys():
                stl = obj_stl[obj_names[i]]["stl"]
                obj_ori = obj_stl[obj_names[i]]["ori"]
            else:
                stl = "bottle.stl"
                obj_ori = [0,0,0,1.0]
            
            color=[0,255,0] if colors is None else colors[i]
            
            markerArray.markers.append(get_marker(
                "package://realsense_3d_detector/meshes/"+stl, obj_poses[i], obj_ori, id=i, scale=[100, 100, 100], color=color))

            # markerArray.markers.append(get_marker(
            #     os.path.join(pkg_path,"meshes/",stl), obj_poses[i], obj_ori, id=i))



        marker_pub.publish(markerArray)



pipe = rs.pipeline()
profile = pipe.start()
cfg = rs.config()

depth_stream=rs.video_stream_profile(profile.get_stream(rs.stream.depth))

intr= depth_stream.get_intrinsics()
print(intr)

align_to = rs.stream.color
align = rs.align(align_to)


# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
# print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale


cfg,weights = model_path+'yolov3.cfg',model_path +'yolov3.weights'
net = cv2.dnn.readNetFromDarknet(cfg,weights)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]


r0=np.zeros((416, 416))
outputs = None
depth_filter=1.0

def trackbar2(x):
    confidence = x/100
    r = r0.copy()
    if outputs is None:
        return
    for output in np.vstack(outputs):
        if output[4] > confidence:
            x, y, w, h = output[:4]
            p0 = int((x-w/2)*416), int((y-h/2)*416)
            p1 = int((x+w/2)*416), int((y+h/2)*416)
            cv2.rectangle(r, p0, p1, 1, 1)
    cv2.imshow('blob', r)
    text = "Bbox confidence={}".format(confidence)
    cv2.displayOverlay('blob', text)

def trackbar3(x):
    global clipping_distance
    clipping_distance = x / (100*depth_scale)



if show:
    cv2.namedWindow("blob", cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('confidence', 'blob', 50, 101, trackbar2)
    cv2.createTrackbar('clipping dist', 'blob', 100, 500, trackbar3)

    trackbar2(50)
    trackbar3(50)


try:
  while not rospy.is_shutdown():
    frameset = pipe.wait_for_frames()

    aligned_frames = align.process(frameset)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    # Validate that both frames are valid
    if not aligned_depth_frame or not color_frame:
        continue

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.flip(np.asanyarray(color_frame.get_data()),2)

    # Remove background - Set pixels further than clipping_distance to grey
    grey_color = 0
    depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
    bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)


    # points = pc.calculate(depth_frame)
    # pc.map_to(color_frame)

    img = color_image.copy()
    
    classes = open(model_path+'coco.names').read().strip().split('\n')
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')


    # determine the output layer
    # print(net.getUnconnectedOutLayers())
    

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    r = blob[0, 0, :, :]


    if show:
        cv2.imshow('blob', r)
        cv2.imshow("aligned", bg_removed)

    text = 'Blob shape={}'.format(blob.shape)
    if show:
        cv2.displayOverlay('blob', text)
    # cv2.waitKey(1)

    net.setInput(blob)
    t0 = time.time()
    outputs = net.forward(ln)
    t = time.time()
    # print('time=', t-t0)


    r0 = blob[0, 0, :, :]
    r = r0.copy()
    if show:
        cv2.imshow('blob', r)

    boxes = []
    confidences = []
    classIDs = []
    h, w = img.shape[:2]

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    obj_names=[]
    obj_poses=[]
    if len(indices) > 0:
        
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            steps=10
            dist_pts = []
            # img_crop = img[x:x + w, y:y:y+h]

            # mask = np.zeros(img_crop.shape[:2],np.uint8)
            # bgdModel = np.zeros((1,65),np.float64)
            # fgdModel = np.zeros((1,65),np.float64)

            # cv2.grabCut(img_crop,mask,(x,y,x+w,y+h),bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
            # mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
            # img_crop = img_crop*mask2[:,:,np.newaxis]

            # cv2.imshow(img_crop)

            # img_crop_removed =  np.where((img_crop_removed > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
            # cropp_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

            img_h,img_w,_=bg_removed.shape
            for x_i in range(max(x,0),min(x+w,img_w),int(w/steps)):
                for y_i in range(max(y,0),min(y+h,img_h),int(h/steps)):
                    if not np.any(bg_removed[y_i,x_i]==grey_color):
                        cv2.circle(img, (x_i,y_i), 2, (0,0,255), 1)
                        # print(aligned_depth_frame.shape)
                        dist_pts.append(aligned_depth_frame.get_distance(x_i, y_i))

            dist = np.mean(dist_pts)

            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f} : {:.2f}m".format(classes[classIDs[i]], confidences[i], dist)
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            obj_names.append(classes[classIDs[i]])
            coords = rs.rs2_deproject_pixel_to_point(intr, [x+w/2, y+h/2], dist)
            # print(coords)
            obj_poses.append(coords)

    if show:
        cv2.imshow('window', img)

    if ros:
        image_pub.publish(cv2_to_imgmsg(img))
        pub_markers(obj_names, obj_poses, colors=colors)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
finally:
    pipe.stop()

cv2.destroyAllWindows()
