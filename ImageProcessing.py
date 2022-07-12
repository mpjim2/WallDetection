from cv2 import dilate
import matplotlib.pyplot as plt
import numpy as np 
import pickle
import cv2 as cv
#from PyIP import ccn


def dilate_erode(frame):
    '''
    
    '''

    kernel = np.ones((7,7),np.uint8)
    small = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(frame, cv.MORPH_OPEN, small)
    opening = cv.morphologyEx(frame, cv.MORPH_CLOSE, small)
    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    sure_fg = cv.erode(opening, kernel, iterations=2)

    return sure_fg, sure_bg 

def combine_red_yellow(frames):
    '''
    
    '''
    frames[0] *= 2
    frames[1] *= 4 

    combined = frames[0] + frames[1]

    return combined

def watershed(fg, unknown, img, CC=True):
    '''
    
    '''
    # Marker labelling

    markers = fg

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    
    # Now, mark the region of unknown with zero
    markers[unknown==1] = 0
    
    markers = cv.watershed(img, markers.astype(np.int32))
    
    
    #img[markers == -1] = [0,0,0]
    
    colors = [[0, 0, 0], 
              [0, 0, 0], 
              [0, 255, 0], 
              [255, 255, 255], 
              [0,0,0], 
              [28, 249, 212], 
              [0, 102, 0], 
              [100,50,0], 
              [50,70,80], 
              [40,40,120], 
              [60,10,250],
              [150,70,20],
              [10,20,250],
              [250,130,110],
              [140,160,120],
              [40,20,90]]

    f = np.zeros(img.shape, np.uint8)
    num_regions = np.unique(markers)
    for i, x in enumerate(num_regions):
        f[markers == x] = colors[i]
    
    
    return f, markers, num_regions

def segment_planes(frame, thresh_mask, cutoff):
    '''
    
    '''

    seg_frames = frame
    frame = np.stack([frame for _ in range(3)], axis=-1)
    
    l = len(seg_frames)
    
    foreground = np.zeros(seg_frames[0].shape, np.uint8)
    bg_c = np.zeros(seg_frames[0].shape, np.uint8)
    
    for i in range(l):
        fg, bg = dilate_erode(seg_frames[i])
        foreground += fg*((i+1)*2)
        bg_c += bg

    #only pixels that are background for all channels are background
    (bg_c == l).astype(np.uint8)

    background = np.logical_or(thresh_mask, bg_c).astype(np.uint8) 
    unknown = cv.subtract(background, (foreground>=1).astype(np.uint8))
    
    #unknown[:cutoff, :] = 0

    img = watershed(foreground, unknown, frame[0].astype(np.uint8), CC=False)

    return img 

    