import math
import time
import cv2 as cv
from cv2 import threshold
import numpy as np
from get_intrinics import intriniscs
import pyrealsense2 as rs
import open3d as o3d


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
    while True:
        # Grab camera data
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        #color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())

        thresh_mask = depth_image <= 100000

        thresh_mask = thresh_mask.astype(np.uint8)

        
        #depth_image = cv.fastNlMeansDenoising(depth_image,None,5,7,5)
        
        
        normals =  estimate_normals(depth_image, K)
        normals = np.nan_to_num(normals)
        
        normals = (normals*255).astype(np.uint8)
        
        #normals_filtered = cv.blur(normals,(13, 13))
        normals_filtered = cv.medianBlur(normals,25)
        normals_filtered = cv.bilateralFilter(normals_filtered, 13, 200, 200)
        
        
        #depth_colormap_f = cv.applyColorMap(cv.convertScaleAbs(depth_image_f, alpha=0.03), cv.COLORMAP_JET)
        #depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)
        #depth_colormap = cv.resize((depth_colormap*255).astype(np.uint8), dsize=(normals.shape[1], normals.shape[0]), interpolation=cv.INTER_AREA)
        
        thresh_mask = cv.resize(thresh_mask, dsize=(normals.shape[1], normals.shape[0]), interpolation=cv.INTER_AREA)
        
        thresh_mask = np.stack([thresh_mask for i in range(3)], axis=-1)
        #normals_vis[1:-1, 1:-1, :] = normals
        
        
        normals*= thresh_mask        
        

        #normals_filtered = (normals_filtered*255).astype(np.uint8)
        normals_filtered*= thresh_mask        

        gray = cv.cvtColor(normals_filtered, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray,200,300)
        
        images = np.hstack((normals, normals_filtered))
        cv.imshow('Normals', images)
        if cv.waitKey(1) == 27:
            break

        
        
    # Stop streaming
    pipeline.stop()