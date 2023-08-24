"""File: Main.py

Entry point for the LaserRunnerRemoval system. 
"""
import numpy as np 
from shapely import Polygon
from ultralytics import YOLO
from laser import IldaLaser
from realsense import RealSense
from cv_utils import find_laser_point
from Tracker import Tracker
import cv2
import time 

def multipoint_calibrate(laser, depth_cam, lsr_pts = None, background_removal=True): 
    """Given a laser and depth camera find the transform between the points 

    Note: The 3D points are currently with respect to the camera, if this where 
    transformed into wrt the laser it would be more accurate. 
    """
    print("Attempting Multipoint Calibration")
    #make sure the laser isn't currently on 
    if lsr_pts is None: 
        #points around the corners of the achievable laser range
        lsr_pts = laser.frame_corners()

    found_lsr_pts = []
    pos_wrt_cam = []
    #should pass in background image if using instead of bool
    if background_removal: 
        background_frame = depth_cam.get_frames()
        background_image = np.asanyarray(background_frame["color"].get_data())
    
    for point in lsr_pts: 
        found_point = None
        attempts = 0
        laser.sendEmpty(x=point[0], y=point[0])
        curr_frame = depth_cam.get_frames()
        while not found_point and attempts<5: 
            laser.add_point(point, pad=False, color = (10, 0, 0), intensity=1)
            laser.sendFrame()
            #handle delay in image accuracy
            for x in range(5): 
                curr_frame = depth_cam.get_frames()
            curr_image = np.asanyarray(curr_frame["color"].get_data())
            if background_removal: 
                image = cv2.absdiff(curr_image, background_image)
            else: 
                image = curr_image
            cv2.imwrite(f"/home/bobby/temp_cal_images/laser_{point[0]}_{point[1]}.jpg", curr_image)
            #cv2.imwrite("/home/bobby/temp_cal_images/background_image.jpg", background_image)
            #cv2.imwrite("/home/bobby/temp_cal_images/diff.jpg", image)
            found_point = find_laser_point(image)
            attempts += 1
        if found_point: 
            print(f"Send Laser Point:{point}")
            print(f"Found Laser Point:{found_point}")
            found_pos = depth_cam.get_pos_location(found_point[0], found_point[1], curr_frame)
            print(f"Found Laser Pos:{found_pos}")
            found_lsr_pts.append(point)
            pos_wrt_cam.append(found_pos)

    laser.sendEmpty()
    #import pdb; pdb.set_trace()
    if len(found_lsr_pts) >= 3:
        pos_wrt_cam = np.array(pos_wrt_cam)
        found_lsr_pts = np.array(found_lsr_pts)     
        #Solve for transform between 3D points 'pos_wrt_cam' and 2D points 'found_lsr_pts'
        # Create an augmented matrix A to solve for the transformation matrix T
        res = np.linalg.lstsq(pos_wrt_cam, found_lsr_pts, rcond=None)
        laser.transform_to_laser = res[0]
        print("----------Calibration Test----------")
        print(f"Sent points: \n{found_lsr_pts}")
        print(f"Calculated points: \n{np.dot(pos_wrt_cam,res[0])}")
    else:  
        print("failed to find at least 3 laser points") 

class MainStateMachine():
    """State Machine that controls what the system is currently doing""" 
    def __init__(self): 
        self.laser = None
        self.model = None
        self.depth_cam = None 
        self.tracker = None 
        self.curr_track = None

    def initialize(self): 
        #Initialize model, camera, laser, and tracker 
        self.laser = IldaLaser()
        self.laser.initialize()
        self.model = YOLO("RunnerSegModel.pt")    
        self.depth_cam = RealSense()
        self.tracker = Tracker()

    def state_acquire(self): 
        #Aquire Image
        frames = self.depth_cam.get_frames()
        cv2.imwrite("/home/bobby/curr_frame.jpg", np.asanyarray(frames["color"].get_data()))

        #Runner detection
        self.background_image = np.asanyarray(frames["color"].get_data())
        res = self.model(self.background_image)

        ## This could move into a preception/model module 
        #create shapely contour from each returned mask 
        if res[0].masks: 
            for cords in res[0].masks.xy:
                polygon = Polygon(cords)
                pos = self.depth_cam.get_pos_location(
                    polygon.centroid.x, 
                    polygon.centroid.y,
                    frames
                )
                if pos is not None: 
                    print(f"track point: {polygon.centroid.x}, {polygon.centroid.y}   pos:{pos}") 
                    track = self.tracker.add_track(pos)

    def state_laser_start(self): 
        if self.laser.transform_to_laser is None:
            calib_pts = self.laser.scale_frame_corners(.15)
            multipoint_calibrate(self.laser, self.depth_cam, lsr_pts = calib_pts)
        #currently only do one runner at at time
        self.curr_track = self.tracker.tracks.pop(0)
        self.laser.add_pos(self.curr_track.pos_wrt_cam, pad=False, color = (0, 0, 255), intensity=255)
        self.laser.sendFrame()

    def state_laser_improve(self): 
        frames = self.depth_cam.get_frames()
        curr_image = np.asanyarray(frames["color"].get_data())
        image = cv2.absdiff(curr_image, self.background_image)
        found_point = find_laser_point(image)
        #Check the distance between the laser and the runner and correct if necessary
        #if np.linalg.norm(found_point, self.curr_track.pos)  

    def state_continous_burn(self):
        time.sleep(1) 
        

if __name__ == "__main__": 
    state_machine  = MainStateMachine()
    state_machine.initialize()
    while True: 
        if state_machine.tracker.has_active_tracks: 
            state_machine.state_laser_start()
            state_machine.state_laser_improve()
            state_machine.state_continous_burn()
        else: 
            #should count number of times no tracks are found, then move forward 
            tracks_found = state_machine.state_acquire()
            



