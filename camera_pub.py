#!/usr/bin/env python

import queue
import rospy 
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from robot_vision.msg import array

from cv_bridge import CvBridge
import pyrealsense2.pyrealsense2 as rs
import numpy as np

import cv2 as cv

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

        config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 6)
        config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 6)

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

    def get_next_frame(self):

        frames = self.pipeline.wait_for_frames()

        frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asarray(color_frame.get_data())

        return color_image, depth_image


def talker():

    pub_color       = rospy.Publisher('/Camera/ColorImage', Image, queue_size=1)
    pub_depth       = rospy.Publisher('/Camera/DepthImage', Image, queue_size=1)
    pub_instrinsics = rospy.Publisher('/Camera/intrinsics', array, queue_size=1 )
    
    rospy.init_node('Image', anonymous=True)
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():

        rospy.loginfo('IDKWID')

        color, depth = cam.get_next_frame()

        #color = cv.cvtColor(color, cv2.COLOR_BGR2GRAY)
        pub_color.publish(br.cv2_to_imgmsg(color))
        pub_depth.publish(br.cv2_to_imgmsg(depth))
        pub_instrinsics.publish(cam.depth_intrinsics)
        
        rate.sleep()


if __name__=='__main__': 
    
    br = CvBridge()
    cam = RealSense_Camera()
    try:
        talker()
    except rospy.ROSInterruptException:
        pass    
