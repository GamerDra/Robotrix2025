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

# Team Name: Toffee Bot
# Team Members: Rudra Ranjit
# Filename:			task.py
# Functions:		control_logic
# 					[ Comma separated list of functions in this file ]
# Global variables:
# 					[ resolution = 512  # pixels fov = 80  # degrees baseline = 0.7  focal_length 

####################### IMPORT MODULES #######################
## You are not allowed to make any changes in this section. ##
##############################################################
import sys
import traceback
import time
import os
import math
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import cv2
import random
import cv2
from move import KalmanFilter3D
from move import ObjectMover


##############################################################

################# ADD GLOBAL VARIABLES HERE ##################

# Camera parameters
resolution = 512  # pixels
fov = 80  # degrees
baseline = 0.7  # m

# Focal length in pixels
focal_length = resolution / (2 * math.tan(math.radians(fov / 2)))
#

##############################################################

################# ADD UTILITY FUNCTIONS HERE #################

def calculate_3d_position(ball_left, ball_right, focal_length, baseline, resolution):
    """
    Calculate the 3D position of the ball relative to the camera and account for real-world scaling.

    Args:
    - ball_left, ball_right: Detected ball positions in the left and right images (x, y, radius).
    - focal_length: Camera focal length in pixels.
    - baseline: Distance between cameras in meters.
    - resolution: Camera resolution (pixels).
    - camera_z_offset: Vertical distance from cameras to the hoop (meters).

    Returns:
    - (X, Y, Z): Real-world position of the ball relative to the camera.
    """
    if ball_left and ball_right:
        x_left, y_left, _ = ball_left
        x_right, y_right, _ = ball_right

        # Calculate disparity
        disparity = abs(x_left - x_right)
        if disparity <= 0:  # Prevent division by zero
            print("Disparity is zero or negative, calculation not possible.")
            return None

        # Calculate depth (Z) in meters
        Z = (focal_length * baseline) / disparity  # Z is in meters

        # Calculate horizontal scaling factor (meters per pixel at depth Z)
        horizontal_scale = baseline / disparity  # Horizontal scaling

        # Principal point (camera center)
        c_x = resolution / 2
        c_y = resolution / 2

        # Calculate X and Y in real-world coordinates
        X = (x_left - c_x) * Z / focal_length
        Y = (y_left - c_y) * Z / focal_length

        return X, Z, -Y
    else:
        print("Ball positions not detected in both images.")
        return None


def get_vision_sensor_images(sim, left_vision_sensor_handle, right_vision_sensor_handle, width=512, height=512, channels=3):
    """
    Retrieve images from two vision sensors using `getVisionSensorImg` and return them.

    Args:
    - sim: RemoteAPIClient object
    - left_vision_sensor_handle: Handle of the left vision sensor
    - right_vision_sensor_handle: Handle of the right vision sensor
    - width: Width of the vision sensor's resolution
    - height: Height of the vision sensor's resolution
    - channels: Number of color channels (default is 3 for RGB)

    Returns:
    - Tuple of numpy arrays: (left_image_array, right_image_array)
    """
    try:
        # Retrieve image and resolution for the left vision sensor
        left_image, left_resolution = sim.getVisionSensorImg(left_vision_sensor_handle)
        left_img_array = np.frombuffer(left_image, dtype=np.uint8).reshape(height, width, channels)

        # Retrieve image and resolution for the right vision sensor
        right_image, right_resolution = sim.getVisionSensorImg(right_vision_sensor_handle)
        right_img_array = np.frombuffer(right_image, dtype=np.uint8).reshape(height, width, channels)
        
        # Flip the images vertically
        left_img_array = cv2.flip(left_img_array, 0)
        right_img_array = cv2.flip(right_img_array, 0)

        # Convert RGB to BGR for OpenCV compatibility if necessary
        if channels == 3:
            left_img_array = cv2.cvtColor(left_img_array, cv2.COLOR_RGB2BGR)
            right_img_array = cv2.cvtColor(right_img_array, cv2.COLOR_RGB2BGR)
        
        return left_img_array, right_img_array

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None



def hoop_position(sim, hoop_handle):
    """
    Get the position of the hoop in the CoppeliaSim scene.

    Args:
    - sim: RemoteAPIClient object
    - hoop_handle: Handle of the hoop object

    Returns:
    - hoop_position: List containing the [x, y, z] position of the hoop
    """
    try:
        hoop_position = sim.getObjectPosition(hoop_handle)  # -1 for world reference frame
        return hoop_position
    except Exception as e:
        print(f"An error occurred while fetching the hoop position: {e}")
        return None
    

    
def visualize_detections(left_img, right_img, ball_left, ball_right):
    """
    Displays the left and right images with ball detections annotated.
    """
    try:
        if ball_left:
            cv2.circle(left_img, (ball_left[0], ball_left[1]), ball_left[2], (0, 255, 0), 2)
        if ball_right:
            cv2.circle(right_img, (ball_right[0], ball_right[1]), ball_right[2], (0, 255, 0), 2)

        # Show images with annotations
        cv2.imshow("Left Image", left_img)
        cv2.imshow("Right Image", right_img)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting visualization loop.")
            raise KeyboardInterrupt  # Trigger exit for the main loop
    except Exception as e:
        print(f"Error in visualization: {e}")




def detect_ball(image):
    """
    Detect the ball in a given image using HSV color segmentation.

    Args:
    - image: Input image (numpy array).

    Returns:
    - Tuple (x, y, radius) of the detected ball's center and radius, or None if no ball is detected.
    """
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the color range for the ball (e.g., orange)
    lower_orange = np.array([5, 100, 100])  # Adjust these values based on the ball's actual color
    upper_orange = np.array([15, 255, 255])  # Adjust these values based on the ball's actual color
    
    # Create a mask for the ball's color
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Perform morphological operations to reduce noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Filter by size (area)
        if cv2.contourArea(contour) > 0:  # Adjust the threshold based on the ball's size
            # Get the center and radius of the enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            # Check if the detected radius is within a reasonable range
            if radius > 1:  # Minimum radius to avoid false positives
                return int(x), int(y), int(radius)
    
    # If no valid ball is found, return None
    return None

def calculate_velocity(prev_pos, current_pos, time_interval):
    """
    Calculate velocity given previous and current positions and the time interval.
    :param prev_pos: Tuple (x, y, z) of the previous position
    :param current_pos: Tuple (x, y, z) of the current position
    :param time_interval: Time interval between the positions
    :return: Velocity as a tuple (vx, vy, vz)
    """
    if prev_pos is None or current_pos is None or time_interval <= 0:
        return None  # Cannot calculate velocity without valid inputs
    
    vx = (current_pos[0] - prev_pos[0]) / time_interval
    vy = (current_pos[1] - prev_pos[1]) / time_interval
    vz = (current_pos[2] - prev_pos[2]) / time_interval
    return (-vx, vy, -vz)

def predict_trajectory(ball_pos, ball_velocity, hoop_z, g=9.81):
    """
    Predict the ball's trajectory and find where it intersects the hoop's Z-plane.

    Args:
    - ball_pos: Tuple (X0, Y0, Z0) representing the ball's current 3D position.
    - ball_velocity: Tuple (Vx, Vy, Vz) representing the ball's velocity in 3D space.
    - hoop_z: Z-coordinate of the hoop plane.
    - g: Gravitational acceleration (default 9.81 m/s^2).

    Returns:
    - Tuple (X_pred, Y_pred): Predicted (X, Y) position where the ball intersects the hoop's plane.
    - time_to_hoop: Time taken for the ball to reach the hoop plane.
    """
    X0, Y0, Z0 = ball_pos
    Vx, Vy, Vz = ball_velocity

    # Calculate time to reach the hoop's Z-plane
    a = -0.5 * g
    b = Vz
    c = Z0 - hoop_z

    # Solve quadratic equation: a*t^2 + b*t + c = 0
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return None, None  # No real solution, ball will not reach the hoop plane

    # Use the positive root for time
    time_to_hoop = (-b - math.sqrt(discriminant)) / (2 * a)
    if time_to_hoop < 0:
        time_to_hoop = (-b + math.sqrt(discriminant)) / (2 * a)

    if time_to_hoop < 0:
        return None, None  # Ball will not reach the hoop plane

    # Predict (X, Y) position at the hoop plane
    X_pred = X0 + Vx * time_to_hoop
    Y_pred = Y0 + Vy * time_to_hoop

    return (X_pred, Y_pred, hoop_z), time_to_hoop

def adjust_hoop_position(sim, hoop_handle, target_position, current_position, max_speed=2):
    """
    Adjust the hoop's position to align with the target position.

    Args:
    - sim: RemoteAPIClient object
    - hoop_handle: Handle of the hoop object
    - target_position: Target position (X, Y, Z) for the hoop
    - current_position: Current position (X, Y, Z) of the hoop
    - max_speed: Maximum speed for the actuators (default is 1.0)

    Returns:
    - None
    """
    try:
        # Calculate the positional errors
        error_x = target_position[0] - current_position[0]
        error_y = target_position[1] - current_position[1]
        error_z = target_position[2] - current_position[2]
        # print(f"Error: {error_x}, {error_y}, {error_z}")
    
        # Normalize the errors to control the speed
        speed_x = max(-max_speed, min(max_speed, error_x))
        speed_y = max(-max_speed, min(max_speed, error_y))
        speed_z = max(-max_speed, min(max_speed, error_z))

        # Set actuator velocities
        joint_x_obj = sim.getObject(r'/basket_bot/actuator_x')
        joint_y_obj = sim.getObject(r'/basket_bot/actuator_y')
        joint_z_obj = sim.getObject(r'/basket_bot/actuator_z')

        sim.setJointTargetVelocity(joint_x_obj, -speed_x)
        sim.setJointTargetVelocity(joint_y_obj, speed_y)
        sim.setJointTargetVelocity(joint_z_obj, -speed_z)
        
        # print(speed_x, speed_y, speed_z)

        # print(f"Adjusting hoop position: Target={target_position}, Current={current_position}")

    except Exception as e:
        print(f"An error occurred while adjusting hoop position: {e}")

def transform_hoop_position(hoop_pos, cam_pos = (-0.35, 0.25, -0.39)):
    # Adjust hoop position based on the camera's location
    adjusted_x = hoop_pos[0] - cam_pos[0]
    adjusted_y = hoop_pos[1] - cam_pos[1]
    adjusted_z = hoop_pos[2] - cam_pos[2]
    return (adjusted_x, adjusted_y, adjusted_z)



##############################################################


def control_logic(sim):
    joint_x_obj = sim.getObject(r'/basket_bot/actuator_x')
    joint_y_obj = sim.getObject(r'/basket_bot/actuator_y')
    joint_z_obj = sim.getObject(r'/basket_bot/actuator_z')    
    sim.setJointTargetVelocity(joint_x_obj, 0)
    sim.setJointTargetVelocity(joint_y_obj, 0.1)
    sim.setJointTargetVelocity(joint_z_obj, 0)

    
    previous_position = None
    # previous_time = time.time()
    previous_time = sim.getSimulationTime()

    try:    
        # Retrieve object handles
        left_vision_sensor_handle = sim.getObject(r'/basket_bot/cam_l')
        right_vision_sensor_handle = sim.getObject(r'/basket_bot/cam_r')
        hoop_handle = sim.getObject(r'/basket_bot/hoop_odom')
        default_hoop_position = hoop_position(sim, hoop_handle)
        print(f"Default Hoop Position: {default_hoop_position}")
        
        # Main control loop
        while sim.getSimulationState() != sim.simulation_stopped:
            coordinates =[[0,0,0]]
            # current_time = time.time()
            current_time = sim.getSimulationTime()
            time_interval = current_time - previous_time

            # Get images from the vision sensors
            left_img, right_img = get_vision_sensor_images(sim, left_vision_sensor_handle, right_vision_sensor_handle)

            if left_img is not None and right_img is not None:
                ball_pos_list = []
                ball_left = detect_ball(left_img)
                ball_right = detect_ball(right_img)

                if ball_left and ball_right:
                    # Fetch hoop position in world coordinates
                    hoop_pos = hoop_position(sim, hoop_handle)
                    # print(f"Hoop Position: {hoop_pos}")
                    if hoop_pos is None:
                        print("Error fetching hoop position")
                        break

                    # Calculate the 3D position of the ball relative to the hoop
                    ball_pos_3d = calculate_3d_position(ball_left, ball_right, focal_length, baseline, resolution)
                    ball_pos_list.append(ball_pos_3d)
                    print(f"Ball 3D Position: {ball_pos_3d}")
                    if not ball_pos_3d:
                        continue
                   
                    
                    # print(f"Actual ball position: {ball_pos_actual}")

                    # Calculate velocity if a previous position exists
                    if previous_position is not None:
                        velocity = calculate_velocity(previous_position, ball_pos_3d, time_interval)
                        # print(f"Ball Velocity: {velocity}")

                        # Predict trajectory
                        hoop_pos = hoop_position(sim, hoop_handle)
                        hoop_pos = transform_hoop_position(hoop_pos)
                        predicted_pos, _ = predict_trajectory(ball_pos_3d, velocity, hoop_pos[2])
                        
                        # print(f"Predicted Ball Position: {predicted_pos}")
                        if predicted_pos:
                            # print(f"Predicted Ball Position at Hoop: {predicted_pos}")
                            for _ in range(12):
                                coordinates.append(ball_pos_3d)
                                np.random.seed(12)
                                dt = 0.1
                                kf = KalmanFilter3D(dt=dt, process_noise_std=0.1, measurement_noise_std=0.5)

                                
                            
                                predicted_positions = []
                                for coord in coordinates:
                                        kf.predict()
                                        predicted_pos = kf.update(coord)
                                        predicted_positions.append(predicted_pos)


                                initial_position = list(ball_pos_3d)  # Starting at (2, 3, 4)
                                target_coordinates = predicted_positions[-1] 
                                
                                mover = ObjectMover(Kp=9.5, Ki=0.1, Kd=1.3, initial_position=initial_position)
                                vx, vy, vz = mover.move_to_target(target_coordinates)
                                sim.setJointTargetVelocity(joint_x_obj, -vx)
                                sim.setJointTargetVelocity(joint_y_obj, vy)
                                sim.setJointTargetVelocity(joint_z_obj, -vz)

                            predicted_positions = np.array(predicted_positions)
                            # adjust_hoop_position(sim, hoop_handle, predicted_pos, hoop_pos)
                            # print("Prediction not possible.")
                        else:
                            hoop_pos = hoop_position(sim, hoop_handle)

                            hoop_pos = list(hoop_pos)
                            hoop_pos[1] = hoop_pos[1] - 1
                            hoop_pos[2] = hoop_pos[2]  - 0.3
                            adjust_hoop_position(sim, hoop_handle, default_hoop_position, hoop_pos) 
                        
                    # Update previous position and time
                    previous_position = ball_pos_3d
                    previous_time = current_time
                    # if hoop_position(sim, hoop_handle)[0] > 4.1:
                    #     sim.setJointTargetVelocity(joint_x_obj, 2)
                    # if abs(hoop_position(sim, hoop_handle)[1]) > 0.3:
                    #     sim.setJointTargetVelocity(joint_y_obj, -1)
                        
                        

                    # Visualize the detections
                    # visualize_detections(left_img, right_img, ball_left, ball_right)

                else:
                    print("Ball not detected in both images.")
                    hoop_pos = hoop_position(sim, hoop_handle)
                    hoop_pos = list(hoop_pos)
                    hoop_pos[1] = hoop_pos[1] - 1
                    hoop_pos[2] = hoop_pos[2] - 0.3
                    target_coordinates = list(hoop_pos)
                    # mover.move_to_target(list(default_hoop_position))
                    # sim.setJointTargetVelocity(joint_x_obj, vx)
                    # sim.setJointTargetVelocity(joint_y_obj, -vy)
                    # sim.setJointTargetVelocity(joint_z_obj, vz)
                    
                    
                    adjust_hoop_position(sim, hoop_handle, default_hoop_position, hoop_pos)
                    
                    

            else:
                print("Error fetching images from vision sensors.")
                break

    except Exception as e:
        print(f"An error occurred during setup: {e}")
    finally:
        cv2.destroyAllWindows()


######### YOU ARE NOT ALLOWED TO MAKE CHANGES TO THE MAIN CODE BELOW #########

if __name__ == "__main__":
	client = RemoteAPIClient()
	sim = client.getObject('sim')	

	try:

		## Start the simulation using ZeroMQ RemoteAPI
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

		## Runs the control logic written by participants
		try:
			control_logic(sim)

		except Exception:
			print('\n[ERROR] Your control_logic function throwed an Exception, kindly debug your code!')
			print('Stop the CoppeliaSim simulation manually if required.\n')
			traceback.print_exc(file=sys.stdout)
			print()
			sys.exit()

		
		## Stop the simulation
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
		## Stop the simulation
		return_code = sim.stopSimulation()
		time.sleep(0.5)
		if sim.getSimulationState() == sim.simulation_stopped:
			print('\nSimulation interrupted by user in CoppeliaSim.')
		else:
			print('\nSimulation could not be interrupted. Stop the simulation manually .')
			sys.exit()