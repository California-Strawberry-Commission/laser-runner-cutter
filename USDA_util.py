"""File USDA_util.py

Utility for controlling a laser based on the feedback from a webcamera. 
"""
import cv2
import PySimpleGUI as sg
from laser import IldaLaser 
import numpy as np
from enum import Enum
import time

def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 6: # if there are more than 5 non working ports stop the testing. 
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
        else:
            is_reading, img = camera.read()
            if is_reading:
                working_ports.append(dev_port)
            else:
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports,working_ports,non_working_ports

class OpMode(Enum): 
    TARGET = 0 
    FIRE = 1

def laser_left(step_size): 
    laser_point[0]=max(0, laser_point[0]-step_size)

def laser_right(step_size): 
    laser_point[0]=laser_point[0]+step_size

def laser_down(step_size): 
    laser_point[1]=laser_point[1]-step_size

def laser_up(step_size): 
    laser_point[1]=laser_point[1]+step_size

#Key mapping 
keymapping = {
    "Left:113":laser_left, 
    "Right:114":laser_right, 
    "Up:111":laser_up, 
    "Down:116":laser_down,  
}

#Information about laser color and on time 
on_time = 0
color = [0, 0, 0]
intensity = 0

# Start capturing video from the webcam
avail_unresp, working, unavail = list_ports()
if len(working) == 0: 
    print("No camera found, check webcam connection")
    cap = None
    camera_num = -1
else: 
    camera_num = working[0]
    cap = cv2.VideoCapture(0)

#Create and initialize a laser object 
laser = IldaLaser()
laser.initialize()

#Information about laser location 
laser_point = np.array(laser.frame_shape)/2

pwm_max = 100
# Define the GUI layout
slider_layout = [
   [sg.Text("Red:"), sg.Slider(range=(0, 255), orientation="h", size=(10, 20), key="-SLIDER_R-", default_value=0)],
   [sg.Text("Blue:"), sg.Slider(range=(0, 255), orientation="h", size=(10, 20), key="-SLIDER_B-", default_value=255)],
   [sg.Text("Green:"), sg.Slider(range=(0, 255), orientation="h", size=(10, 20), key="-SLIDER_G-", default_value=0)],
   [sg.Text("Intensity:"), sg.Slider(range=(0, 255), orientation="h", size=(10, 20), key="-SLIDER_I-", default_value=255)],
   [sg.Text("Pulse Width Modulation:"), sg.Slider(range=(1, pwm_max), orientation="h", size=(10, 20), key="-SLIDER_PWM-", default_value=pwm_max)]
]
time_layout = [
    [sg.Text("On Time (Seconds)"), sg.Text(size=(10,1), key="-ON_TIME-")],
    [sg.InputText("1", key="-TIME_INPUT-", size=(20, 1))],
    [sg.Button("Start Test")]
]
setup_layout = [
    [sg.Text("Tracking Laser Step Size"), sg.Slider(range=(1,20), orientation="h", size=(10, 20), key="-SLIDER_RESOLUTION-", default_value=5)],
    [sg.Text("Camera Number:"), sg.Combo(working, font=('Arial Bold', 14),  expand_x=True, enable_events=True,  readonly=False, key='-CAM_NUM-', default_value=0)],
]

layout = [
    [
        [sg.Image(filename="", key="-IMAGE-")],
        [   
            sg.Column(setup_layout), 
            sg.Column(slider_layout), 
            sg.Column(time_layout),
        ], 
        [sg.Button("Exit")]
    ]
]

# Create the GUI window
window = sg.Window("Webcam GUI", layout, return_keyboard_events=True, finalize=True)

mode = OpMode.TARGET
start_ts = time.time()
while True:
    event, values = window.read(timeout=20)

    if event == sg.WINDOW_CLOSED or event == "Exit":
        break

    if event=='-CAM_NUM-': 
        cam_num = values['-CAM_NUM-']
        cap = cv2.VideoCapture(cam_num)

    if cap is not None: 
        ret, frame = cap.read()  # Read a frame from the webcam

    # Display the captured frame on the GUI
    if ret:
        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["-IMAGE-"].update(data=imgbytes)

    if mode==OpMode.FIRE:
        curr_ts = time.time()
        if curr_ts - start_ts > on_time: 
            laser.sendEmpty(x=laser_point[0], y=laser_point[1])
            mode=OpMode.TARGET
        continue

    # Check for key presses using PySimpleGUI's built-in key event handling
    if event in keymapping.keys(): 
        step_size = int(values['-SLIDER_RESOLUTION-'])
        keymapping[event](step_size)  

    if event == "Start Test": 
        laser.sendEmpty(x=laser_point[0], y=laser_point[1])
        on_time = float(values["-TIME_INPUT-"])
        color[0] = int(values['-SLIDER_R-'])
        color[1] = int(values['-SLIDER_G-'])
        color[2] = int(values['-SLIDER_B-'])
        intensity = int(values['-SLIDER_I-'])
        pwm = int(values['-SLIDER_PWM-'])
        start_ts = time.time()
        pad = (pwm_max-pwm)
        laser.add_point(laser_point, color, pad=pad, intensity=intensity)
        mode=OpMode.FIRE
        laser.sendFrame()

    if mode == OpMode.TARGET: 
        laser.add_point(laser_point, (7, 0, 0), pad=0, intensity=1)
        laser.sendFrame()
   

    focus =  window.find_element_with_focus()
    if focus: 
        if focus.Type == "combo": 
            window.TKroot.focus_force()

    

# Release the webcam and close the OpenCV window
cap.release()
window.close()
