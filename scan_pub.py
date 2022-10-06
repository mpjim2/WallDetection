#!/usr/bin/env python

import queue
import rospy 
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from robot_vision.msg import array, points

#from cv_bridge import CvBridge
import pyrealsense2.pyrealsense2 as rs
import numpy as np



class RealSense_Camera:

    def __init__(self):

        #initialize Camera
        self.pipeline = rs.pipeline()
        config = rs.config()

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        profile = self.pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
    
        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
        # Get stream profile and camera intrinsics
        profile = self.pipeline.get_active_profile()
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        depth_intrinsics = depth_profile.get_intrinsics()

        fx, fy = depth_intrinsics.fx, depth_intrinsics.fy
        cx, cy = depth_intrinsics.ppx, depth_intrinsics.ppy
        
        self.depth_intrinsics = [fx, fy, cx, cy]
        self.pointcloud = rs.pointcloud()   
    

    def depth_to_scan(self, depth_frame, height):
        
        points = self.pointcloud.calculate(depth_frame)
        points = points.get_vertices()
        points = np.asanyarray(points).view(np.float32).reshape(-1, 3)

        #remove points that are too far away
        filter_far = np.where(points[:, 2] < 1)
        points = points[filter_far]

        filter_far = np.where(abs(points[:, 0]) < 0.8)
        points = points[filter_far]

        #remove very close points
        filter_close = np.where(np.invert(((points[:, 2] <=0.3) & (abs(points[:, 0]) <=0.2))))
        points = points[filter_close]

        #filter 'line' in pointcloud to get 'laser scan' equivalent
        scan = np.where(abs(points[:, 1]) < 0.003)
        points = points[scan]

        return points[:, 0],  points[:, 2] 

    def get_next_frame(self):

        frames = self.pipeline.wait_for_frames()

        frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        
        color_frame = frames.get_color_frame()
        
        return color_frame, depth_frame


def talker():

    
    pub_scan = rospy.Publisher('/Camera/PC_Scan',points, queue_size=1 )
    
    rospy.init_node('Image', anonymous=True)
    rate = rospy.Rate(1)

    while not rospy.is_shutdown():

        rospy.loginfo('PointCloud Scan published')

        color, depth = cam.get_next_frame()
        points_X, points_Y = cam.depth_to_scan(depth, 0)
        
        point_msg = points()
        point_msg.X = points_X
        point_msg.Y = points_Y

        pub_scan.publish(point_msg)
        rate.sleep()


if __name__=='__main__': 
    
    #br = CvBridge()
    cam = RealSense_Camera()
    try:
        talker()
    except rospy.ROSInterruptException:
        pass    