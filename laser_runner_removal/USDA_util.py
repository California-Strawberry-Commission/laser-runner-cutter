"""File USDA_util.py

Utility for controlling a laser based on the feedback from a webcamera. 
"""
import cv2
import PySimpleGUI as sg
from laser import IldaLaser 
import numpy as np
from enum import Enum
import time
from datetime import datetime

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
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                working_ports.append(dev_port)
            else:
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports,working_ports,non_working_ports

def safety_check(): 
    layout = [
        [sg.Text("This will enable a high power laser,\n Ensure safety requirements are followed.")],
        [sg.Button("Enable Laser"), sg.Button("Cancel")]
    ]
    
    window = sg.Window("Safety Check", layout, finalize=True)
    
    while True:
        event, _ = window.read(timeout=50)
        
        if event in (sg.WIN_CLOSED, "Cancel"):
            window.close()
            return False
        elif event == "Enable Laser":
            window.close()
            return True
        
       

def idle_check(): 
    start_time = time.time()
    layout = [
        [sg.Text("No user input detected.")],
        [sg.Button("Continue Operating"), sg.Button("Disable Laser")]
    ]
    
    window = sg.Window("Idle Check", layout, finalize=True)
    
    
    while time.time() - start_time < 10:
        event, _ = window.read(timeout=50)
        
        if event in (sg.WIN_CLOSED, "Disable Laser"):
            window.close()
            return False
        elif event == "Continue Operating":
            window.close()
            return True
    window.close()
    return False
        

class OpMode(Enum): 
    LASER_OFF = 0
    TARGET = 1
    PRE_FIRE = 2
    FIRE = 3
    POST_FIRE = 4

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
    camera_num = working[-1]
    cap = cv2.VideoCapture(camera_num)
    ret, frame = cap.read()
    size = [frame.shape[1], frame.shape[0]]

#Create and initialize a laser object 
laser = IldaLaser()
laser.initialize()

#Information about laser location 
laser_point = np.array(laser.frame_shape)/2

#pwm_max = 100
# Define the GUI layout
slider_layout = [
   [sg.Text("Red:"), sg.Slider(range=(0, 255), orientation="h", size=(10, 20), key="-SLIDER_R-", default_value=0)],
   [sg.Text("Blue:"), sg.Slider(range=(0, 255), orientation="h", size=(10, 20), key="-SLIDER_B-", default_value=255)],
   [sg.Text("Green:"), sg.Slider(range=(0, 255), orientation="h", size=(10, 20), key="-SLIDER_G-", default_value=0)],
   #Disabled for now
   #[sg.Text("Intensity:"), sg.Slider(range=(0, 255), orientation="h", size=(10, 20), key="-SLIDER_I-", default_value=255)],
   #[sg.Text("Pulse Width Modulation:"), sg.Slider(range=(1, pwm_max), orientation="h", size=(10, 20), key="-SLIDER_PWM-", default_value=pwm_max)]
   [sg.Button("Exit")],
]
time_layout = [
    [sg.Text("Camera Number:"), sg.Combo(working,  expand_x=True, enable_events=True,  readonly=False, key='-CAM_NUM-', default_value=camera_num)],
    [sg.Text("On Time (Seconds)"), sg.Text(size=(10,1), key="-ON_TIME-")],
    [sg.InputText("1", key="-TIME_INPUT-", size=(20, 1))],
    [sg.Button("Start Test")],
    [sg.Text("Laser Off", key='-Laser State-', font=("Helvetica", 14))]
]
setup_layout = [
    [sg.Text("Laser Position Control, directional buttons can also be used")], 
    [sg.Button("LEFT"), sg.Button("UP"), sg.Button("DOWN"), sg.Button("RIGHT")],
    [sg.Text("Tracking Laser Step Size"), sg.Slider(range=(1,20), orientation="h", size=(10, 20), key="-SLIDER_RESOLUTION-", default_value=5)],
    [sg.Button("Enable Tracking Laser")],
]
safety_layout = [
    [sg.Image('LaserWarning.png', expand_x=True, expand_y=True)],
    [sg.Text("WARNING - High Power Laser", font=("Helvetica", 14))],
    [sg.Text("Make sure no one is in the path of the laser. \nNever look directly at the laser without eye protection. \nMake sure no unintended materials will be hit by the laser. \nTurn the laser off completely after us.")],
]
filename_layout = [
    [sg.Text("All tests are stored in the Videos folder. \nDefault name is the time in the format \nYearMonthDayHourMinuteSecond \nOptionally a different name can be input below.")], 
    [sg.InputText("", key="-FILENAME_INPUT-", size=(20, 1))]    
]

layout = [
    [
        [sg.Image(filename="", key="-IMAGE-"), sg.Column(safety_layout)],
        [   
            sg.Column(setup_layout), 
            sg.Column(slider_layout), 
            sg.Column(time_layout),
            sg.Column(filename_layout),
        ], 
    ]
]

# Create the GUI window
window = sg.Window("Webcam GUI", layout, return_keyboard_events=True, finalize=True)
laser_safety_conf = False

mode = OpMode.LASER_OFF
last_action_ts = None
default_timeout = 60
while True:     
    #Disable the laser if no action has been preformed within the default timeout
    if last_action_ts: 
        if time.time() - last_action_ts>default_timeout: 
            if not idle_check(): 
                laser_safety_conf = False
                mode = OpMode.LASER_OFF
                laser.sendEmpty(x=laser_point[0], y=laser_point[1])
                last_action_ts = None
            else: 
                last_action_ts = time.time()

    event, values = window.read(timeout=50)

    if event == sg.WINDOW_CLOSED or event == "Exit":
        break 
    
    if not event == "__TIMEOUT__":
        last_action_ts = time.time() 

    ret=None
    if event=='-CAM_NUM-': 
        cam_num = values['-CAM_NUM-']
        cap = cv2.VideoCapture(cam_num)
        ret, frame = cap.read()  # Read a frame from the webcam
        size = [frame.shape[1], frame.shape[0]]

    elif cap is not None: 
        ret, frame = cap.read()  # Read a frame from the webcam

    # Display the captured frame on the GUI
    if ret:
        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["-IMAGE-"].update(data=imgbytes)

    if mode == OpMode.PRE_FIRE: 
        if time.time() > prefire_ts + 3: 
            start_ts = time.time()
            laser.sendFrame()
            mode = OpMode.FIRE

    if mode==OpMode.FIRE:
        curr_ts = time.time()
        if curr_ts - start_ts > on_time: 
            laser.sendEmpty(x=laser_point[0], y=laser_point[1])
            last_action_ts = time.time()
            mode=OpMode.POST_FIRE
            postfire_ts = time.time()
            window['-Laser State-'].update("Test Completed")
    
    if mode==OpMode.POST_FIRE: 
        curr_ts = time.time() 
        if curr_ts - 3 > postfire_ts: 
            mode=OpMode.LASER_OFF   

    if mode == OpMode.FIRE or mode == OpMode.PRE_FIRE or mode == OpMode.POST_FIRE: 
        save_ts = time.time()
        rec.write(frame)
        
    # Check for key presses using PySimpleGUI's built-in key event handling
    if event in keymapping.keys(): 
        step_size = int(values['-SLIDER_RESOLUTION-'])
        keymapping[event](step_size)  
    if event == "UP": 
        step_size = int(values['-SLIDER_RESOLUTION-'])
        laser_up(step_size)
    if event == "DOWN": 
        step_size = int(values['-SLIDER_RESOLUTION-'])
        laser_down(step_size)
    if event == "RIGHT": 
        step_size = int(values['-SLIDER_RESOLUTION-'])
        laser_right(step_size)
    if event == "LEFT": 
        step_size = int(values['-SLIDER_RESOLUTION-'])
        laser_left(step_size)

    if event == "Enable Tracking Laser": 
        if not laser_safety_conf: 
            laser_safety_conf = safety_check()
        if laser_safety_conf: 
            laser_on_time = time.time()
            mode = OpMode.TARGET
            window['-Laser State-'].update("Tracking Laser On")

    if event == "Start Test": 
        if not laser_safety_conf: 
            laser_safety_conf = safety_check()
        if laser_safety_conf: 
            window['-Laser State-'].update("Test Ongoing")
            laser.sendEmpty(x=laser_point[0], y=laser_point[1])
            on_time = float(values["-TIME_INPUT-"])
            color[0] = int(values['-SLIDER_R-'])
            color[1] = int(values['-SLIDER_G-'])
            color[2] = int(values['-SLIDER_B-'])
            #disabled for now
            #intensity = int(values['-SLIDER_I-'])
            #pwm = int(values['-SLIDER_PWM-'])
            #pad = (pwm_max-pwm)
            pad = 0
            laser.add_point(laser_point, color, pad=pad, intensity=intensity)
            prefire_ts = time.time()
            filename = values['-FILENAME_INPUT-']
            if filename: 
                rec_name = f'/home/bobby/Videos/{filename}.avi'
            else: 
                datetime_obj=datetime.fromtimestamp(prefire_ts)
                datetime_string=datetime_obj.strftime( "%Y%m%d%H%M%S" )
                rec_name = f'/home/bobby/Videos/{datetime_string}.avi'
            
            rec = cv2.VideoWriter(
                rec_name, 
                0, 
                20.0, 
                size
            )
            mode=OpMode.PRE_FIRE            

    if mode == OpMode.TARGET: 
        laser.add_point(laser_point, (10, 0, 0), pad=0, intensity=1)
        laser.sendFrame()

    focus =  window.find_element_with_focus()
    if focus: 
        if focus.Type == "combo": 
            window.TKroot.focus_force()

    

# Release the webcam and close the OpenCV window
cap.release()
window.close()