"""Class for using a realsense depth camera"""

import pyrealsense2 as rs
import numpy as np 
class RealSense():
    def __init__(self): 
        #Setup code based on https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)
        self.setup_inten_extrins()
        
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
   
        self.depthMinMeters = .1
        self.depthMaxMeters = 10
        

    def get_clipping_distance(self): 
        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
        clipping_distance_in_meters = 1 #1 meter
        self.clipping_distance = clipping_distance_in_meters / depth_scale

    def get_frames(self): 
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame() 
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None
        
        return {"color": color_frame, "depth": depth_frame}

    def setup_inten_extrins(self): 
        color_prof = self.profile.get_stream(rs.stream.color)
        depth_prof = self.profile.get_stream(rs.stream.depth)

        self.depth_intrins = depth_prof.as_video_stream_profile().get_intrinsics()
        self.color_intrins = color_prof.as_video_stream_profile().get_intrinsics()        
        print(self.color_intrins)
        self.depth_to_color_extrinsic = depth_prof.get_extrinsics_to(color_prof)
        self.color_to_depth_extrinsic = color_prof.get_extrinsics_to(depth_prof)

    def color_pixel_to_depth(self, x, y, frame):
        """Given the location of a x-y point in the color frame, return the corresponding x-y point in the depth frame."""
        #Based of a number of realsense github issues including
        #https://github.com/IntelRealSense/librealsense/issues/5440#issuecomment-566593866
        print(f"color point: [{x} {y}]")
        depth_pixel = rs.rs2_project_color_pixel_to_depth_pixel(frame["depth"].get_data(), self.depth_scale, self.depthMinMeters, self.depthMaxMeters, self.depth_intrins, self.color_intrins, self.depth_to_color_extrinsic, self.color_to_depth_extrinsic, (x, y))
        print(f"depth point: {depth_pixel}")
        if depth_pixel[0] < 0 or depth_pixel[1] < 0: 
            return None
        return depth_pixel

    def get_pos_location(self, x, y, frame): 
        """Given an x-y point in the color frame, return the x-y-z point with respect to the camera"""  
       depth_point = self.color_pixel_to_depth(x, y, frame)
        if not depth_point: 
            return None
        depth = frame["depth"].get_distance(round(depth_point[0]), round(depth_point[1]))
        pos_wrt_color = rs.rs2_deproject_pixel_to_point(self.color_intrins, [x, y], depth)
        pos_wrt_color = np.array(pos_wrt_color)*1000
        print(f"color_pos: {pos_wrt_color}") 
        return pos_wrt_color