import math
import time
import cv2 as cv
from cv2 import threshold
import numpy as np
import pyrealsense2 as rs
import open3d as o3d

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

def normalization(data):
   mo_chang =np.sqrt(np.multiply(data[:,:,0],data[:,:,0])+np.multiply(data[:,:,1],data[:,:,1])+np.multiply(data[:,:,2],data[:,:,2]))
   mo_chang = np.dstack((mo_chang,mo_chang,mo_chang))
   return data/mo_chang

def estimate_normals(img1_depth, K):

    height, width = img1_depth.shape
    x,y=np.meshgrid(np.arange(0,width),np.arange(0,height))
    x=x.reshape([-1])
    y=y.reshape([-1])
    xyz=np.vstack((x,y,np.ones_like(x)))
    pts_3d=np.dot(np.linalg.inv(K),xyz*img1_depth.reshape([-1]))
    pts_3d_world=pts_3d.reshape((3,height,width))
    f= pts_3d_world[:,1:height-1,2:width]-pts_3d_world[:,1:height-1,1:width-1]
    t= pts_3d_world[:,2:height,1:width-1]-pts_3d_world[:,1:height-1,1:width-1]
    
    normal_map=np.cross(f,t,axisa=0,axisb=0)
    normal_map=normalization(normal_map)
    normal_map=normal_map*0.5+0.5
    #alpha = np.full((height-2,width-2,1), (1.), dtype="float32")
    #normal_map=np.concatenate((normal_map,alpha),axis=2)

    return normal_map


def intrinsic_matrix(depth_intrinsics):
    
    w, h = depth_intrinsics.width, depth_intrinsics.height

    fx, fy = depth_intrinsics.fx, depth_intrinsics.fy

    cx, cy = depth_intrinsics.ppx, depth_intrinsics.ppy

    matrix = np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]])
    return matrix


def normalize_img(data):
    
    data_min = np.min(data, axis=(1,2), keepdims=True)
    data_max = np.max(data, axis=(1,2), keepdims=True)

    data = (data - data_min) / (data_max - data_min)
    return data


def gridOfImages(images):
    
    grid = np.block([[images[0], images[1], images[2]],
                    [images[3], images[4], np.zeros(images[0].shape)]])

    return grid

if __name__ == '__main__':

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
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

    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # Get stream profile and camera intrinsics
    profile = pipeline.get_active_profile()
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()

    K = intrinsic_matrix(depth_intrinsics)
    n_clusters = 5
    init_clusters = None
    while True:
        # Grab camera data
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        #color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())

        
        #depth_image = cv.fastNlMeansDenoising(depth_image,None,5,7,5)
        
        
        normals =  estimate_normals(depth_image, K)
        normals = np.nan_to_num(normals)
        

        #threshold mask to filter out pixels that are far away and thus noisy
        thresh_mask = depth_image <= 1000
        thresh_mask = thresh_mask.astype(np.uint8)
        thresh_mask = cv.resize(thresh_mask, dsize=(normals.shape[1], normals.shape[0]), interpolation=cv.INTER_AREA)

        normals *= np.stack([thresh_mask for i in range(3)], axis=-1)
        normals_flat = normals.reshape((normals.shape[0] * normals.shape[1], 3))
        
        #1 cluster per plane + 1 for out of range pixels
        
        if init_clusters is None:
            kmu = KMeans(n_clusters = n_clusters, random_state=0).fit(normals_flat)
        else:
            kmu = KMeans(n_clusters=n_clusters, init=init_clusters).fit(normals_flat)
        

        init_clusters = kmu.cluster_centers_

        labels = kmu.labels_.reshape((normals.shape[0], normals.shape[1]))
        
        
        cluster_imgs = [labels == i for i in range(n_clusters)]
        
        images = gridOfImages(cluster_imgs).astype(np.uint8) * 255
        
        
        cv.imshow('clusters', images)

        if cv.waitKey(1) == 27:
            break

        
        
    # Stop streaming
    pipeline.stop()