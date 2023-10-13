"""File: cv_utils.py

Computer vision utilities. 
"""
import cv2
import numpy as np
import time 

SAVE_DIR = None

def find_laser_point(frame): 
    height, width, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret,mask = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)

    if SAVE_DIR: 
        ts = time.time()
        cv2.imwrite(f"{SAVE_DIR}/{ts}_grayscale.jpg", gray)    
        cv2.imwrite(f"{SAVE_DIR}/{ts}_mask.jpg", mask)
    return check_circularity(mask)
    
def find_laser_point_red(frame):
    """Find red laser points using the red minus max(green, blue) frame."""
    height, width, _ = frame.shape
    rg = frame[:, :, 2].astype(np.float32) - np.max(np.dstack((frame[:, :, 0], frame[:, :, 1])), axis=2).astype(np.float32)

    max_val = np.max(rg)
    min_val = np.min(rg)
    cur_range = max_val - min_val
    scaling_factor = 255 / cur_range
    norm_rg = (rg - min_val) * scaling_factor
    norm_rg = norm_rg.astype(np.uint8)
    
    # Create a mask for red regions
    ret,mask = cv2.threshold(norm_rg,150,255,cv2.THRESH_BINARY)

    if SAVE_DIR: 
        ts = time.time()
        cv2.imwrite(f"{SAVE_DIR}/{ts}_normrg.jpg", norm_rg)    
        cv2.imwrite(f"{SAVE_DIR}/{ts}_mask.jpg", mask)

    return check_circularity(mask)
    
def check_circularity(mask): 
    """Given a binary mask find contours and return the centroid of the more circular contour"""
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate centers of the contours (green rings)
    max_circularity = 0
    if not contours: 
        return []  
   
    best_contour = None 
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area < 50: 
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity > max_circularity: 
            best_contour = contour
            max_circularity = circularity
        
    if best_contour is None: 
        return []
        
    # Compute the moments of the contour
    M = cv2.moments(best_contour)
    if M["m00"] != 0:
        # Calculate the center of the contour
        cX = M["m10"] / M["m00"]
        cY = M["m01"] / M["m00"]
        return [(cX, cY)]
    else: 
        return []
