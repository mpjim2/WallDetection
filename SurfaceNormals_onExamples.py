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

    depth_img = 