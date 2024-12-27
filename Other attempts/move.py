
import time
import numpy as np
from scipy.linalg import block_diag

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp  # Proportional gain
        self.Ki = Ki  # Integral gain
        self.Kd = Kd  # Derivative gain
        
        self.prev_error = 0
        self.integral = 0
    
    def update(self, error, delta_time):
        self.integral += error * delta_time
        derivative = (error - self.prev_error) / delta_time if delta_time > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

class ObjectMover:
    def __init__(self, Kp, Ki, Kd, initial_position):
        self.pid_x = PIDController(Kp, Ki, Kd)
        self.pid_y = PIDController(Kp, Ki, Kd)
        self.pid_z = PIDController(Kp, Ki, Kd)
        self.position = initial_position  # Custom initial position (x, y, z)
    
    def move_to_target(self, target, delta_time=0.1, tolerance=0.01):
        final_vx, final_vy, final_vz = 0, 0, 0  # Initialize final velocities
        
        error_x = target[0] - self.position[0]
        error_y = target[1] - self.position[1]
        error_z = target[2] - self.position[2]
            
            # Update velocities based on PID
        final_vx = self.pid_x.update(error_x, delta_time)
        final_vy = self.pid_y.update(error_y, delta_time)
        final_vz = self.pid_z.update(error_z, delta_time)
            
            # Update position based on velocities
        self.position[0] += final_vx * delta_time
        self.position[1] += final_vy * delta_time
        self.position[2] += final_vz * delta_time
            # Check if within tolerance
            
        time.sleep(delta_time)
        
        return final_vx, final_vy, final_vz
# Define the Kalman Filter class
class KalmanFilter3D:
		def __init__(self, dt, process_noise_std, measurement_noise_std):
			self.dt = dt
			
			# State vector [x, y, z, vx, vy, vz]
			self.x = np.zeros((6, 1))
			
			# State transition matrix (constant velocity model)
			self.F = np.array([
				[1, 0, 0, dt, 0, 0],
				[0, 1, 0, 0, dt, 0],
				[0, 0, 1, 0, 0, dt],
				[0, 0, 0, 1, 0, 0],
				[0, 0, 0, 0, 1, 0],
				[0, 0, 0, 0, 0, 1]
			])
			
			# Measurement matrix (we can only measure positions)
			self.H = np.array([
				[1, 0, 0, 0, 0, 0],
				[0, 1, 0, 0, 0, 0],
				[0, 0, 1, 0, 0, 0]
			])
			
			# Process noise covariance
			q = process_noise_std**2
			self.Q = block_diag(np.eye(3) * q * (dt**2), np.eye(3) * q)
			
			# Measurement noise covariance
			r = measurement_noise_std**2
			self.R = np.eye(3) * r
			
			# Estimation uncertainty covariance
			self.P = np.eye(6) * 1000
			
		def predict(self):
			self.x = np.dot(self.F, self.x)
			self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
			return self.x[:3].flatten()
		
		def update(self, z):
			"""
			Update the filter with a new measurement.
			Parameters:
				z (array-like): Measured coordinates [x, y, z].
			"""
			z = np.reshape(z, (3, 1))
			y = z - np.dot(self.H, self.x)
			S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
			K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
			self.x = self.x + np.dot(K, y)
			self.P = self.P - np.dot(K, np.dot(self.H, self.P))
			return self.x[:3].flatten()