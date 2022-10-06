from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
import actionlib
import actionlib_tutorials.msg

from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
import numpy as np
import json
import rospy
import os


def obj_piramid(file, pos, ori, n=3, id0=0, spacing=[0.05125, 0.05125, 0.125]):
    marker_list = []
    for i in range(n):
        for j in range(n-i):
            for k in range(n-i-j):
                x = pos[0] - (j * spacing[1]+i*spacing[0]/2)
                y = pos[1] + (k * spacing[0]+j*spacing[1]/2 + i*spacing[0]/2)
                z = (i+0.5) * spacing[2]
                marker_list.append(get_marker(
                    file, [x, y, z], ori, id0+i*n*n+j*n+k, scale=[1.25, 1.25, 1.0]))
    return marker_list


def get_marker(file, pos, ori, id=0, scale=[1, 1, 1], color=[0,255,0], text=''):
    marker = Marker()

    marker.header.frame_id = "/camera"
    marker.header.stamp = rospy.Time.now()

    # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3; text: 9
    marker.type = 9#marker.MESH_RESOURCE
    marker.id = id

    # Set the scale of the marker
    marker.scale.x = 0.001*scale[0]
    marker.scale.y = 0.001*scale[1]
    marker.scale.z = 0.001*scale[2]

    # Set the color
    marker.color.r = color[2]/255.0
    marker.color.g = color[1]/255.0
    marker.color.b = color[0]/255.0
    marker.color.a = 1.0

    marker.text = text


    marker.pose.position.x = pos[0]
    marker.pose.position.y = pos[1]
    marker.pose.position.z = 0 if len(pos) == 2 else pos[2]

    marker.pose.orientation.x = ori[0]
    marker.pose.orientation.y = ori[1]
    marker.pose.orientation.z = ori[2]
    marker.pose.orientation.w = ori[3]

    marker.action = marker.ADD

    # marker.mesh_use_embedded_materials = True
    # marker.mesh_resource = file
    return marker


def clean_marker():
    marker = Marker()
    marker.id = 0
    marker.action = Marker.DELETEALL
    return marker