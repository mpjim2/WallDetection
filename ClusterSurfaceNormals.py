from encodings import normalize_encoding
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

import sys

import warnings


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BLACK="\[\033[0;30m\]"

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


def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm


def clusterNormals():


    return 


def initialize():
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

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
   
    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    # Get stream profile and camera intrinsics
    profile = pipeline.get_active_profile()
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()

    K = intrinsic_matrix(depth_intrinsics)
    
    return pipeline, K, align

def threshold_mask(depth_image, out_shape):
    #threshold mask to filter out pixels that are far away and thus noisy
    mask = depth_image <= 2000
    #filter out "bad" pixels on the left side of the image
    #mask[:, :25] = 0
    #filter out top part of the image
    #mask[:250, :] = 0
    
    return cv.resize(mask.astype(np.uint8), dsize=out_shape, interpolation=cv.INTER_AREA)

def planeNormal(normals, markers, m):

    relevant_normals = normals[markers==m] 
    wall_normal = np.median(relevant_normals, axis=0)

    return normalize(wall_normal)

if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    #writer= cv.VideoWriter('basicvideo.mp4', cv.VideoWriter_fourcc(*'DIVX'), 20, (478,1276))

    pipeline, K, align = initialize()

    n_clusters = 4

    processing_steps = []
    # Configure depth and color streams
    init_clusters = None
    while True:
        # Grab camera data
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        frames = align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asarray(color_frame.get_data())
        

        processing_steps.append(depth_image)
        normals =  estimate_normals(depth_image, K)
        normals = np.nan_to_num(normals)


        processing_steps.append(normals)
        thresh_mask = threshold_mask(depth_image, out_shape=(normals.shape[1], normals.shape[0]))
        #flatten threshold mask to only cluster in range pixels
        thresh_mask_flat = np.where(thresh_mask.reshape((thresh_mask.shape[0] * thresh_mask.shape[1])))
        
        print("Look at these SHAPES")
        print(thresh_mask.shape, normals.shape)
        #CLUSTERING
        normals *= np.stack([thresh_mask for i in range(normals.shape[-1])], axis=-1)
        normals_flat = normals.reshape((normals.shape[0] * normals.shape[1], normals.shape[-1]))
        
        points2cluster = normals_flat[thresh_mask_flat]

    
        #1 cluster per plane + 1 for out of range pixels
        if init_clusters is None:
            kmu = KMeans(n_clusters = n_clusters, random_state=0).fit(points2cluster)
        else:
            kmu = KMeans(n_clusters=n_clusters, init=init_clusters).fit(points2cluster)
        

        init_clusters = kmu.cluster_centers_
        #Clustering End

        #Segmentation
        left_wall = np.argmin([np.linalg.norm(i - [-1, 0, 0]) for i in init_clusters])
        floor     = np.argmin([np.linalg.norm(i - [0, 1, 0]) for i in init_clusters])      
        
        labels = make_label_img([list(kmu.labels_)], thresh_mask_flat, (normals.shape[0], normals.shape[1]), n_clusters)
        
        cluster_imgs = [(labels == i).astype(np.uint8) for i in range(n_clusters)]
        
        cluster_left  = [(labels == left_wall).astype(np.uint8)]
        cluster_floor = [(labels == floor).astype(np.uint8)]
        colored_left  = coloredClusters(cluster_left)

        processing_steps.append(colored_left)

        segmented_L, markers_L, num_regions_L = IP.segment_planes(cluster_left, (thresh_mask == 0).astype(np.uint8), cutoff=250)


        segmented_F, markers_F, num_regions_F = IP.segment_planes(cluster_floor, (thresh_mask == 0).astype(np.uint8), cutoff=250)
        
        processing_steps.append(segmented_L)
        #markers+=1
        #num_labels, labels_im = cv.connectedComponents(markers.astype(np.uint8))
       
        left_normal = planeNormal(normals, markers_L, num_regions_L[-1])
        floor_normal = planeNormal(normals, markers_F, num_regions_F[-1])


        control_vector = normalize(np.cross(left_normal, floor_normal))

        colored_floor =  coloredClusters(cluster_imgs)

        color_image = cv.resize(color_image, dsize=(normals.shape[1], normals.shape[0]), interpolation=cv.INTER_AREA)

        markers_F==num_regions_F[-1]
        markers_L==num_regions_L[-1]
        
        idk = np.logical_xor(markers_F==num_regions_F[-1],markers_L==num_regions_L[-1])

        color_image[np.logical_and(markers_L==num_regions_L[-1], idk)] = [0,255,0]
        #color_image[np.logical_and(markers_F==num_regions_F[-1], idk)] = [255, 0, 0]

        processing_steps.append(color_image)


        #combo = np.hstack((colored, colored_onlyFlorr))
        #writer.write(combined)
        cv.putText(color_image, str(np.round(control_vector, 2)),(100,100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        cv.imshow('Segmented', color_image)

        if cv.waitKey(1) == 27:
            break

        
    #writer.release()
    # Stop streaming
    pipeline.stop()