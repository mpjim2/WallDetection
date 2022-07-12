from encodings import normalize_encoding
import math
import time
import cv2 as cv
from cv2 import threshold
import numpy as np
from ClusterSurfaceNormals import planeNormal, threshold_mask
import pyrealsense2 as rs
import open3d as o3d

from sklearn.cluster import KMeans, SpectralClustering

import ImageProcessing as IP 
import matplotlib.pyplot as plt

import sys

import warnings

def intrinsic_matrix(depth_intrinsics):
    
    w, h = depth_intrinsics.width, depth_intrinsics.height

    fx, fy = depth_intrinsics.fx, depth_intrinsics.fy

    cx, cy = depth_intrinsics.ppx, depth_intrinsics.ppy

    matrix = np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]])
    return matrix

def normalization(data):
   mo_chang =np.sqrt(np.multiply(data[:,:,0],data[:,:,0])+np.multiply(data[:,:,1],data[:,:,1])+np.multiply(data[:,:,2],data[:,:,2]))
   mo_chang = np.dstack((mo_chang,mo_chang,mo_chang))
   return data/mo_chang

def make_label_img(labels, thresh_idcs, out_shape, n_clusters):

    out_shape = (out_shape[1], out_shape[0])
    label_img = np.ones(out_shape[0] * out_shape[1]) * (n_clusters+1)
    label_img[thresh_idcs[0]] = labels

    label_img = label_img.reshape((out_shape))
    return label_img

def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

def planeNormal(normals, markers, m):

    relevant_normals = normals[markers==m] 
    wall_normal = np.median(relevant_normals, axis=0)

    return normalize(wall_normal)


class RGBD_WallFollower:


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

        self.K = intrinsic_matrix(depth_intrinsics)

        self.init_clusters = None
        self.n_clusters    = 4

        self.followed_wall = None


    def get_next_frame(self):

        frames = self.pipeline.wait_for_frames()

        frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asarray(color_frame.get_data())

        return color_image, depth_image

    def compute_normals(self, depth):
        warnings.filterwarnings("ignore")
        height, width = depth.shape
        x,y=np.meshgrid(np.arange(0,width),np.arange(0,height))
        x=x.reshape([-1])
        y=y.reshape([-1])
        xyz=np.vstack((x,y,np.ones_like(x)))
        pts_3d=np.dot(np.linalg.inv(self.K),xyz*depth.reshape([-1]))
        pts_3d_world=pts_3d.reshape((3,height,width))
        f= pts_3d_world[:,1:height-1,2:width]-pts_3d_world[:,1:height-1,1:width-1]
        t= pts_3d_world[:,2:height,1:width-1]-pts_3d_world[:,1:height-1,1:width-1]
        
        normal_map=np.cross(f,t,axisa=0,axisb=0)
        normal_map=normalization(normal_map)
        #normal_map=normal_map*0.5+0.5
        # alpha = np.full((height-2,width-2,1), (1.), dtype="float32")
        # normal_map=np.concatenate((normal_map,alpha),axis=2)

        return np.nan_to_num(normal_map)

    def threshold_mask(self, depth_image, out_shape):
        
        #threshold mask to filter out pixels that are far away and thus noisy
        mask = depth_image <= 2000
        #filter out "bad" pixels on the left side of the image
        #mask[:, :25] = 0
        #filter out top part of the image
        #mask[:250, :] = 0

        return cv.resize(mask.astype(np.uint8), dsize=out_shape, interpolation=cv.INTER_AREA) 
    
    
    def cluster_normals(self, normals, thresh_mask):

        normals *= np.stack([thresh_mask for _ in range(normals.shape[-1])], axis=-1)
        normals_flat = normals.reshape((normals.shape[0] * normals.shape[1], normals.shape[-1]))
        
        thresh_mask_flat = np.where(thresh_mask.reshape((thresh_mask.shape[0] * thresh_mask.shape[1])))

        points2cluster = normals_flat[thresh_mask_flat]

        #1 cluster per plane + 1 for out of range pixels
        if self.init_clusters is None:
            kmu = KMeans(n_clusters = self.n_clusters, random_state=0).fit(points2cluster)
        else:
            kmu = KMeans(n_clusters=self.n_clusters, init=self.init_clusters).fit(points2cluster)
        

        self.init_clusters = kmu.cluster_centers_

        return kmu

    def segmentation(self, clustering, target_plane, thresh_mask, out_shape):
        
        #Segmentation
        wall = np.argmin([np.linalg.norm(i - target_plane) for i in self.init_clusters])
        
        thresh_mask_flat = np.where(thresh_mask.reshape((thresh_mask.shape[0] * thresh_mask.shape[1])))

        labels = make_label_img([list(clustering.labels_)], thresh_mask_flat, out_shape, self.n_clusters)
                
        binary_wall  = [(labels == wall).astype(np.uint8)]

        segmented, markers, num_regions = IP.segment_planes(binary_wall, (thresh_mask == 0).astype(np.uint8), cutoff=250)
        
        return (segmented, markers, num_regions)


    def compute_control_vector(self):

        frames = self.get_next_frame()

        normals = self.compute_normals(frames[1])

        out_shape = (normals.shape[1], normals.shape[0])
        threshold_mask = self.threshold_mask(frames[1], out_shape)

        clustering = self.cluster_normals(normals, threshold_mask)

        #if no wall is selected for following choose that is more prevalent in the image (left or right)
        if self.followed_wall is None:
            
            cur_max = 0
            for candidate in [[1, 0, 0], [-1, 0 ,0]]:

                s, m, nr = self.segmentation(clustering, candidate, threshold_mask, out_shape)
                
                area = np.count_nonzero(m == nr[-1])
                if area > cur_max:
                    self.followed_wall = candidate
                    wall_s, wall_m, wall_nr = s, m, nr 
                    cur_max = area

            floor_s, floor_m, floor_nr = self.segmentation(clustering, [0,1,0], threshold_mask, out_shape)

            if self.followed_wall == [1, 0, 0]:
                print('Following Right Wall')
            else:
                print("Following Left Wall")

        else: 
            
            wall_s, wall_m, wall_nr = self.segmentation(clustering, self.followed_wall, threshold_mask, out_shape)
            floor_s, floor_m, floor_nr = self.segmentation(clustering, [0,1,0], threshold_mask, out_shape)

        wall_normal  = planeNormal(normals, wall_m, wall_nr[-1])
        floor_normal = planeNormal(normals, floor_m, floor_nr[-1])

        return normalize(np.cross(wall_normal, floor_normal))


if __name__=='__main__':

    WF = RGBD_WallFollower()

    vec = WF.compute_control_vector()
    print(vec)  