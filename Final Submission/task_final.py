'''
*****************************************************************************************
*
*        =================================================
*                    IEEE Robotrix 2024-25
*        =================================================
*
*  This script is intended for implementation of the final hackaton
*  task of IEEE Robotrix 2024-25
*
*****************************************************************************************
'''

# Team Name:		Toffee Bot
# Team Members:		1. Rudra Gandhi, 2. Ranjit Tanneru
# Filename:			task.py
# Functions:		control_logic, calculate_intrinsic_matrix, calculate_extrinsic_matrix, image_preprocess, detect_ball_color, estimate_3d_position
# 					[ Comma separated list of functions in this file ]
# Global variables:	resolution_x, resolution_y, fov, baseline, fov_rad, focal_length, cx, cy
# 					[ List of global variables defined in this file ]

####################### IMPORT MODULES #######################
## You are not allowed to make any changes in this section. ##
##############################################################
import sys
import traceback
import time
import os
import math
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import cv2
import random
import numpy as np

##############################################################

################# ADD GLOBAL VARIABLES HERE ##################
resolution_x = 512 # pixels
resolution_y = 512 # pixels
fov = 80 # degrees
baseline = 0.7 # m
fov_rad = np.deg2rad(fov) # fov in radians
focal_length = resolution_x / (2 * np.tan(fov_rad / 2)) # focal length

# Principal point (cx, cy)
cx = resolution_x / 2
cy = resolution_y / 2

##############################################################

################# ADD UTILITY FUNCTIONS HERE #################
def calculate_intrinsic_matrix():
    '''
    A simple function to calculate the intrinsic matrix of the given camera
    '''
    intrinsic_matrix = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0,  0,  1]
    ])
    return intrinsic_matrix

def calculate_extrinsic_matrix():
    '''
    A simple function to calculate the extrinsic matrix of the given camera
    '''
    R = np.eye(3)  # Identity rotation matrix
    T = np.array([baseline, 0, 0])  # Baseline along the X-axis
    extrinsic_matrix = np.hstack((R, T.reshape(3, 1)))
    return extrinsic_matrix

def preprocess_image(sim, vision_sensor_handle, height=512, width=512, channels=3):
    '''
    Extract and preprocess the image from a vision sensor in CoppeliaSim.
    '''
    image, resolution = sim.getVisionSensorImg(vision_sensor_handle)
    img_array = np.frombuffer(image, dtype=np.uint8).reshape(height, width, channels)
    img_array = cv2.flip(img_array, 0)  # Flip vertically
    if channels == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    return img_array

def detect_ball_color(image):
    '''
    Detect the basketball in the image using color-based segmentation.
    '''
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([15, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
    
        moments = cv2.moments(largest_contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            return cx, cy
    return None, None  # Return None if no valid basketball is detected

class PID:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0

    def update(self, current_value, dt):
        error = self.setpoint - current_value
        P = self.Kp * error
        self.integral += error * dt
        I = self.Ki * self.integral
        derivative = (error - self.prev_error) / dt
        D = self.Kd * derivative
        self.prev_error = error
        return P + I + D

def estimate_3d_position(ball_left_x, ball_right_x, ball_left_y):
    '''
    Estimate the 3D position of the basketball using stereo vision.
    '''
    disparity = ball_left_x - ball_right_x
    if disparity != 0:
        Z = (focal_length * baseline) / disparity
        X = Z * (ball_left_x - cx) / focal_length
        Y = Z * (ball_left_y - cy) / focal_length
        return X, Y, Z
    return None, None, None
##############################################################

def control_logic(sim):
    '''
    Main control logic for the basketball-catching robot.
    '''
    joint_x_obj = sim.getObject(r'/basket_bot/actuator_x')
    joint_y_obj = sim.getObject(r'/basket_bot/actuator_y')
    joint_z_obj = sim.getObject(r'/basket_bot/actuator_z')
    left_vision_sensor_handle = sim.getObject(r'/basket_bot/cam_l')
    right_vision_sensor_handle = sim.getObject(r'/basket_bot/cam_r')
    rail = sim.getObject(r'/basket_bot/rail_2')

    sim.setJointTargetVelocity(joint_x_obj, 0)
    sim.setJointTargetVelocity(joint_y_obj, 0)
    sim.setJointTargetVelocity(joint_z_obj, 0)
    start_time = 0

    intrinsic_matrix = calculate_intrinsic_matrix()
    extrinsic_matrix = calculate_extrinsic_matrix()

    pid_x = PID(5, 0.1, 1)
    pid_y = PID(2, 0.1, 1)
    pid_z = PID(5, 0.1, 1)

    while True:
        try:
            left_img = preprocess_image(sim, left_vision_sensor_handle)
            right_img = preprocess_image(sim, right_vision_sensor_handle)

            ball_left_x, ball_left_y = detect_ball_color(left_img)
            ball_right_x, _ = detect_ball_color(right_img)
            rail_position = sim.getObjectPosition(rail)

            if ball_left_x is not None and ball_right_x is not None:
                X, Y, Z = estimate_3d_position(ball_left_x, ball_right_x, ball_left_y)
                if X is not None:
                    print(f"3D Position: X={X:.2f}, Y={Y:.2f}, Z={Z:.2f}")

                    x_error = X - 0.31
                    y_error = Z - 1
                    z_error = Y - 0.11

                    dt = 0.1
                    end_time=time.time()

                    x_control = pid_x.update(x_error, dt)
                    y_control = pid_y.update(y_error, dt)
                    z_control = pid_z.update(z_error, dt)

                    # print(x_error)
                    # print(x_control,y_control,z_control)


                    sim.setJointTargetVelocity(joint_x_obj, x_control)
                    if end_time-start_time > 11 or y_error < 4:
                        sim.setJointTargetVelocity(joint_y_obj, -y_control)
                    else:
                        sim.setJointTargetVelocity(joint_y_obj, -2+rail_position[1])
                    sim.setJointTargetVelocity(joint_z_obj, -z_control)
                    # sim.setJointTargetVelocity(joint_x_obj, 0)
                    # sim.setJointTargetVelocity(joint_y_obj, 0)
                    # sim.setJointTargetVelocity(joint_z_obj, 0)

            else:
                print("balls")
                sim.setJointTargetVelocity(joint_x_obj, rail_position[0])
                sim.setJointTargetVelocity(joint_y_obj, -4)
                sim.setJointTargetVelocity(joint_z_obj, -0.3 - rail_position[2])
                start_time = time.time()

            cv2.imshow("Left Camera", left_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except KeyboardInterrupt:
            print("Simulation interrupted by user.")
            break

    cv2.destroyAllWindows()
    return None

######### YOU ARE NOT ALLOWED TO MAKE CHANGES TO THE MAIN CODE BELOW #########

if __name__ == "__main__":
    client = RemoteAPIClient()
    sim = client.getObject('sim')

    try:
        try:
            return_code = sim.startSimulation()
            if sim.getSimulationState() != sim.simulation_stopped:
                print('\nSimulation started correctly in CoppeliaSim.')
            else:
                print('\nSimulation could not be started correctly in CoppeliaSim.')
                sys.exit()
        except Exception:
            print('\n[ERROR] Simulation could not be started !!')
            traceback.print_exc(file=sys.stdout)
            sys.exit()

        try:
            control_logic(sim)
        except Exception:
            print('\n[ERROR] Your control_logic function threw an Exception, kindly debug your code!')
            print('Stop the CoppeliaSim simulation manually if required.\n')
            traceback.print_exc(file=sys.stdout)
            print()
            sys.exit()

        try:
            return_code = sim.stopSimulation()
            time.sleep(0.5)
            if sim.getSimulationState() == sim.simulation_stopped:
                print('\nSimulation stopped correctly in CoppeliaSim.')
            else:
                print('\nSimulation could not be stopped correctly in CoppeliaSim.')
                sys.exit()
        except Exception:
            print('\n[ERROR] Simulation could not be stopped !!')
            traceback.print_exc(file=sys.stdout)
            sys.exit()

    except KeyboardInterrupt:
        return_code = sim.stopSimulation()
        time.sleep(0.5)
        if sim.getSimulationState() == sim.simulation_stopped:
            print('\nSimulation interrupted by user in CoppeliaSim.')
        else:
            print('\nSimulation could not be interrupted. Stop the simulation manually.')
            sys.exit()