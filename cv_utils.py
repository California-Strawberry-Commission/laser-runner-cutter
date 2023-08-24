"""File: cv_utils.py

Computer vision utilities
"""
import cv2
import numpy as np

def find_centers(frame):
    """Find green laser points using the green minus max(red, blue) frame."""
    #Find the green minus max(red, blue) frame
    g_max = frame[:, :, 1].astype(np.float32) - np.max(np.dstack((frame[:, :, 0], frame[:, :, 2])), axis=2).astype(np.float32)

    #Normalize based on the green max frame
    max_val = np.max(g_max)
    min_val = np.min(g_max)
    cur_range = max_val - min_val
    scaling_factor = 255 / cur_range
    norm_g_max = (g_max - min_val) * scaling_factor
    norm_g_max = norm_g_max.astype(np.uint8)
    
    # Create a mask for green regions
    cv2.imwrite("/home/bobby/gr.jpg", norm_g_max)
    ret,mask = cv2.threshold(norm_g_max,150,255,cv2.THRESH_BINARY)
    cv2.imwrite("/home/bobby/mask.jpg", mask)

    #Optional Morphologic operations 
    #kernel = np.ones((5,5),np.uint8)
    #mask = cv2.erode(mask,kernel,iterations = 2)
    #mask = cv2.dilate(mask,kernel,iterations = 2)
    #cv2.imwrite("/home/bobby/mask_modified.jpg", mask)

    # Find contours in the mask
    return check_circularity(mask)


def find_laser_point(frame): 
    height, width, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("/home/bobby/gray.jpg", gray)
    ret,mask = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
    cv2.imwrite("/home/bobby/mask.jpg", mask)
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
    #mask = cv2.adaptiveThreshold(norm_gr,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,101,2)
    cv2.imwrite("/home/bobby/rg.jpg", norm_rg)
    ret,mask = cv2.threshold(norm_rg,150,255,cv2.THRESH_BINARY)
    cv2.imwrite("/home/bobby/mask.jpg", mask)

    return check_circularity(mask)
    
def check_circularity(mask): 
    """Given a binary mask find contours and return the centroid of the more circular contour"""
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate centers of the contours (green rings)
    max_circularity = 0
    if not contours: 
        return None  
   
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area < 50: 
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity > max_circularity: 
            best_contour = contour
            max_circularity = circularity
        
    # Compute the moments of the contour
    M = cv2.moments(best_contour)
    print(cv2.contourArea(best_contour))
    if M["m00"] != 0:
        # Calculate the center of the contour
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)
    else: 
        return None
