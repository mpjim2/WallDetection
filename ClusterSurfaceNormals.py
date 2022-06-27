import math
import time
import cv2 as cv
from cv2 import threshold
import numpy as np
import pyrealsense2 as rs
import open3d as o3d

from sklearn.cluster import KMeans, SpectralClustering

import ImageProcessing as IP 
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
    #normal_map=normal_map*0.5+0.5
    # alpha = np.full((height-2,width-2,1), (1.), dtype="float32")
    # normal_map=np.concatenate((normal_map,alpha),axis=2)

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

def coloredClusters(images):

    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255,255,255], [0,0,0], [28, 249, 212], [0, 102, 0]]
    colored = np.zeros((images[0].shape[0],images[0].shape[1], 3))
    for i in range(len(images)):
        t = np.full(colored.shape, colors[i])
        colored += t * np.stack([images[i] for _ in range(t.shape[-1])], axis=-1) 

    return colored


def make_label_img(labels, thresh_idcs, out_shape, n_clusters):

    label_img = np.ones(out_shape[0] * out_shape[1]) * (n_clusters+1)
    label_img[thresh_idcs[0]] = labels

    label_img = label_img.reshape((out_shape))
    return label_img

def fiveDimNormals(normals):
    
    ret = []
    w, h, _ = normals.shape

    for i in range(w):
        for j in range(h):
            x,y,z = normals[i,j]
            ret.append((x,y,z,i,j))

    return np.array(ret)

if __name__ == '__main__':


    #writer= cv.VideoWriter('basicvideo.mp4', cv.VideoWriter_fourcc(*'DIVX'), 20, (478,1276))


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
    n_clusters = 4
    init_clusters = None
    while True:
        # Grab camera data
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asarray(color_frame.get_data())
        

        #depth_image = cv.blur(depth_image, (5,5))
        normals =  estimate_normals(depth_image, K)
        normals = np.nan_to_num(normals)
        
        
        #normals = cv.medianBlur(normals, 3)
        #normals = cv.blur(normals, (3,3))
        #normals = cv.GaussianBlur(normals, (5, 5), 255)

        #threshold mask to filter out pixels that are far away and thus noisy
        thresh_mask = depth_image <= 1600
        thresh_mask = thresh_mask.astype(np.uint8)
        thresh_mask = cv.resize(thresh_mask, dsize=(normals.shape[1], normals.shape[0]), interpolation=cv.INTER_AREA)


        
        normals *= np.stack([thresh_mask for i in range(normals.shape[-1])], axis=-1)
        normals_flat = normals.reshape((normals.shape[0] * normals.shape[1], normals.shape[-1]))
        
        #normas_flat_w_coords = fiveDimNormals(normals)

        
        #flatten threshold mask to only cluster in range pixels
        thresh_mask_flat = np.where(thresh_mask.reshape((thresh_mask.shape[0] * thresh_mask.shape[1])))
        
        points2cluster = normals_flat[thresh_mask_flat]

    
        #1 cluster per plane + 1 for out of range pixels
        if init_clusters is None:
            kmu = KMeans(n_clusters = n_clusters, random_state=0).fit(points2cluster)
        else:
            kmu = KMeans(n_clusters=n_clusters, init=init_clusters).fit(points2cluster)
        

        init_clusters = kmu.cluster_centers_
        
        print(init_clusters)
        labels = make_label_img([list(kmu.labels_)], thresh_mask_flat, (normals.shape[0], normals.shape[1]), n_clusters)
        
        
        cluster_imgs = [(labels == i).astype(np.uint8) for i in range(n_clusters)]
        
        #segmented = IP.segment_planes(cluster_imgs, (thresh_mask == 0).astype(np.uint8))


        #images = gridOfImages(cluster_imgs)
        colored =  coloredClusters(cluster_imgs)
        
        plt.imshow(colored)
        plt.show()
        #color_image = cv.resize(color_image, dsize=(normals.shape[1], normals.shape[0]), interpolation=cv.INTER_AREA).astype(np.uint8)

        #combined = np.hstack([(normals * 255), colored, segmented]).astype(np.uint8)
        #writer.write(combined)
        
        # cv.imshow('clusters', colored)

        # if cv.waitKey(1) == 27:
        #     break

        
    #writer.release()
    # Stop streaming
    pipeline.stop()