from sklearn.cluster import KMeans, SpectralClustering


import queue
import rospy 
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from robot_vision.msg import array

from walknet_curvewalking.msg import robot_control

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

def normal_mask(markers, m): 

    return markers == m
    
def planeNormal(normals, mask):

    relevant_normals = normals[mask] 
    wall_normal = np.median(relevant_normals, axis=0)

    return normalize(wall_normal)


class RGBD_WallFollower:

    def __init__(self):
        
        print('INITILIZATION')
        self.K = None

        self.init_clusters = None
        self.n_clusters    = 4

        self.followed_wall = None
        
        self.control_vec = None
        self.br = CvBridge()

        self.F = None
        self.W = None 

        self.current_speed = 0.1
        self.current_dir = 0.0

    def set_intrinsics(self, data):

        self.K = intrinsic_matrix(*data.data)
    
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

    def compute_control_angle(self, vector):

        self.current_dir = math.atan2(vector[0], vector[2])
        
        sys.stdout.write(f"\r {bcolors.BOLD}{bcolors.OKBLUE} Control Vector: {bcolors.OKGREEN}%2.5f , {bcolors.OKGREEN}%2.5f , {bcolors.OKGREEN}%2.5f Angle: %2.5f" %(self.control_vec[0], self.control_vec[1], self.control_vec[2], self.current_dir))
        sys.stdout.flush()

        # if not rospy.is_shutdown():
        #     msg = robot_control()
        #     msg.speed_fact = self.current_speed
        #     msg.pull_angle = 0
        #     rospy.loginfo("JoyController publish msg: " + str(msg))
        #     pub.publish(msg)


    def compute_control_vector(self, data):
        
        
        depth = self.br.imgmsg_to_cv2(data)

        normals = self.compute_normals(depth)

        out_shape = (normals.shape[1], normals.shape[0])
        threshold_mask = self.threshold_mask(depth, out_shape)

        clustering = self.cluster_normals(normals, threshold_mask)

        cluster_counts = np.unique(clustering.labels_, return_counts=True)
        x = sorted(cluster_counts[0], key= lambda k: cluster_counts[1][k], reverse=True)
        
        #if no wall is selected for following choose that is more prevalent in the image (left or right)
        if self.followed_wall is None:

            for cluster_idx in x: 
                
                cluster_center = clustering.cluster_centers_[cluster_idx]
                s, m, nr = self.segmentation(clustering, cluster_center, threshold_mask, out_shape)
                pn   = planeNormal(normals, (m==nr[-1]))
                area = np.count_nonzero(m == nr[-1])           

                if pn[1] <= 0.1:
                    if area >= 50000:
                        wall_s, wall_m, wall_nr, wall_normal = s, m, nr, pn
                        break
        else: 
            
            wall_s, wall_m, wall_nr = self.segmentation(clustering, self.followed_wall, threshold_mask, out_shape)
        
        floor_s, floor_m, floor_nr = self.segmentation(clustering, [0,1,0], threshold_mask, out_shape)

        self.W = normal_mask(wall_m, wall_nr[-1])
        self.F = normal_mask(floor_m, floor_nr[-1])

        self.W[self.F==True] = False

        floor_normal = planeNormal(normals, self.F)
        wall_normal  = planeNormal(normals, self.W)

        self.control_vec = normalize(np.cross(wall_normal, floor_normal))
        self.compute_control_angle(self.control_vec)

        

    def show_view(self, RGB):
        
        frame = self.br.imgmsg_to_cv2(RGB)

        if not self.W is None:
            frame = cv.resize(frame, dsize=(self.W.shape[1], self.W.shape[0]), interpolation=cv.INTER_AREA) 

            frame[np.nonzero(self.F)] = [255, 0, 0]
            frame[np.nonzero(self.W)] = [0, 255, 0]
            
            cv.imshow('robot_view', frame)
            cv.waitKey(1) 

def intrinsic_matrix(fx, fy, cx, cy):

    matrix = np.array([[fx, 0 , cx],
                       [0 , fy, cy],
                       [0 , 0 , 1 ]])
    return matrix


def callback_depth(data):
    br = CvBridge()
    image = br.imgmsg_to_cv2(data)
    

if __name__=='__main__':

    WF = RGBD_WallFollower()
    rospy.init_node('listener', anonymous=True)
    
    #pub = rospy.Publisher('/control_robot', robot_control, queue_size=1)

    rospy.Subscriber('/Camera/intrinsics', array, WF.set_intrinsics)

    rospy.Subscriber('/Camera/DepthImage', Image, WF.compute_control_vector)
    
    rospy.Subscriber('/Camera/ColorImage', Image, WF.show_view)
    rospy.spin()
