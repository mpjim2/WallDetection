#!/usr/bin/env python

from sklearn import linear_model
import scipy.spatial as spat

import pyrealsense2 as rs
import queue
import rospy 

from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from robot_vision.msg import array, points

from sensor_msgs.msg import Joy
from enum import Enum

import cv2 as cv
from cv_bridge import CvBridge

import numpy as np

from encodings import normalize_encoding
import math
import time
import cv2 as cv
from cv2 import threshold
import numpy as np

import robot_vision.ImageProcessing as IP 
from robot_vision.ClusterSurfaceNormals import planeNormal, threshold_mask
from walknet_curvewalking.msg import robot_control


import matplotlib.pyplot as plt

import sys

import warnings


def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm   


class PS4ButtonMapping(Enum):
    CROSS = 0
    SQUARE = 3
    TRIANGLE = 2
    CIRCLE = 1
    R1 = 5
    R2 = 7
    L1 = 4
    L2 = 6
    SHARE = 8
    OPTIONS = 9
    PS = 10
    JOYLEFT = 11
    JOYRIGHT = 12

class PS4AxesMapping(Enum):
    DPADY = 7
    DPADX = 6
    JOYLEFTX = 0
    JOYLEFTY = 1
    JOYRIGHTX = 3
    JOYRIGHTY = 4
    R2 = 2
    L2 = 5


class WallFollower:

    def __init__(self):

        self.br = CvBridge()

        self.cur_model = None
        self.followed_wall  = None

        self.global_counter = 0

        self.pub = rospy.Publisher('/control_robot', robot_control, queue_size=1)

        self.RIGHT_DEF = None

        self.line_points = None
        self.control_angle = 0
        self.speed = 0
        

        rospy.on_shutdown(self.zeroMsg)

    def compute_ctrl_angle(self):

        v = self.line_points[1] - self.line_points[0]
        a = math.atan2(v[1], v[0])

        #turn right makes sure the robot does not get stuck in corners
        if self.RIGHT_DEF is None:
            if np.sign(a) >= 0:
                a = a - math.pi/2
                self.RIGHT_DEF = True
            else:
                a = math.pi/2 + a
                
                self.RIGHT_DEF = False
        else:
            if self.RIGHT_DEF:
                if np.sign(a) >= 0:
                    rospy.loginfo("Wall to the Left; angle OK")
                    a = a - math.pi/2
                else:
                    a = -1.57
                    rospy.loginfo("Wall to the Left; angle negative")
            else:
                if np.sign(a +math.pi/2) <= 0:
                    rospy.loginfo("Wall to the Right; angle OK " +str(a))
                    a = math.pi/2 + a
                else:
                    a = 1.57
                    rospy.loginfo("Wall to the Right; Angle positive")
        
        rospy.loginfo("\t " + str(a))
        return a

    def joy_control(self, data):
        
        if data.buttons[PS4ButtonMapping.TRIANGLE.value] == 1:
            self.speed = 0.04
            
        elif data.buttons[PS4ButtonMapping.SQUARE.value] == 1:
            self.speed = 0
        else:
            return
    def sequential_RANSAC(self, X, Y):
        
        done = False
        sets = []
        ransacs = []
        origin_dist = []
        point_dist = []
        angles = []

        min_points = []
        while not done:
            
            ransacs.append(linear_model.RANSACRegressor(loss="squared_error", residual_threshold=0.001))
            ransacs[-1].fit(X, Y)
            
            inlier_mask = ransacs[-1].inlier_mask_
            sets.append((X[inlier_mask], Y[inlier_mask]))

            #Compute minimal distance to Wall from origin(robot position)
            setpoints = np.hstack([X[inlier_mask], np.expand_dims(Y[inlier_mask], axis=1)])

            distance  = spat.distance.cdist(setpoints, np.array([[0,0]]))
            origin_dist.append(distance.min())
            idx = np.argmin(distance)

            min_points.append([setpoints[idx]])    
        
            #compute minimal distance to previously followed wall
            if self.followed_wall is not None:

                distance_p = spat.distance.cdist(setpoints, self.followed_wall.reshape(1,2))
                point_dist.append(distance_p.min())
                idx = np.argmin(distance_p)
                min_points[-1].append(setpoints[idx])

            angle = math.atan(ransacs[-1].estimator_.coef_)
            angles.append(angle)

            out = np.logical_not(inlier_mask)

            X = X[out]
            Y = Y[out]
            
            if X.shape[0] < 50:
                done = True
        
        return sets, origin_dist, angles, point_dist, min_points, ransacs


    def compute_control_vector(self, data):

        X = np.expand_dims(np.array(data.X), axis=1)
        Y = np.array(data.Y)
        
        approach_perpendicular = False
        
        if X.shape[0] >= 100:
            
            self.global_counter += 1
            sets, origin_dist, angles, point_dist, min_points, ransacs = self.sequential_RANSAC(X, Y)
            #No Wall found yet -> Choose Closest
            if self.followed_wall is None:
                wall_idx = np.argmin(origin_dist)
                wall_set = sets[wall_idx]
                
                self.followed_wall = np.array(min_points[wall_idx])[0]
                
            else: 
                
                wall_idx = np.argmin(point_dist)
            
                wall_set = sets[wall_idx]
                self.followed_wall = min_points[wall_idx][1]
                
            
            if self.followed_wall is not None:

                line_X = np.linspace(wall_set[0].min(), wall_set[0].max(), num=100)[:, np.newaxis]
                ransac_line = ransacs[wall_idx].predict(line_X)

            #Save Start and Endpoint of followed Wall
                if self.line_points is None:
                    sp = np.array([line_X[0][0], ransac_line[0]])    
                    ep = np.array([line_X[-1][0], ransac_line[-1]])

                    self.line_points = np.array([sp, ep])

                else:

                    sp = np.array([line_X[0][0], ransac_line[0]])    
                    ep = np.array([line_X[-1][0], ransac_line[-1]])

                    
                    s_id = np.argmin(spat.distance.cdist(np.array([sp, ep]), self.line_points[0].reshape(1,2)))
                    
                    if s_id == 0:
                        self.line_points = np.array([sp, ep])
                    else:
                        self.line_points = np.array([ep, sp])
                
                if abs(origin_dist[wall_idx]) <= 0.8:
                    #turn when wall is close enough                    
                    self.control_angle = self.compute_ctrl_angle()
                    #if too close to wall trun full circle

                    if abs(origin_dist[wall_idx]) <= 0.5: 
                        if self.RIGHT_DEF:
                            factor = -1
                        else:
                            factor = +1
                        self.control_angle += factor * 1
                        rospy.loginfo("Too Close to wall turn by" + str(self.control_angle))

                    elif abs(origin_dist[wall_idx]) >= 0.7:                        
                        if self.RIGHT_DEF:
                            factor = +1
                        else:
                            factor = -1
                        self.control_angle += factor * 0.5
                        rospy.loginfo("Too far from wall turn by" + str(self.control_angle))
                        
                    self.cur_model = ransacs[wall_idx]
                    
            else:
                self.publish_control(0)
        else:
            self.control_angle = 0
        
        self.publish_control()

    def zeroMsg(self):
        msg = robot_control()
        msg.speed_fact = 0
        msg.pull_angle = 0
        self.pub.publish(msg)

    def publish_control(self):
        
        #rospy.loginfo("Control Angle " + str(self.control_angle) + str(self.RIGHT_DEF))
        if self.control_angle < -1.57:
            self.control_angle = -1.57
        if self.control_angle > 1.57:
            self.control_angle = 1.57

        
        if not rospy.is_shutdown():
            msg = robot_control()
            msg.speed_fact = self.speed
            msg.pull_angle = self.control_angle            
            self.pub.publish(msg)
        

def listen():

    rospy.init_node('Wall-Follower', anonymous=True)

    rospy.Subscriber('/Camera/PC_Scan', points, WF.compute_control_vector)
    
    rospy.Subscriber('/wxmark4/joy', Joy, WF.joy_control)
    rospy.spin()

if __name__=='__main__':
    warnings.filterwarnings("ignore")
    WF = WallFollower()
    listen()