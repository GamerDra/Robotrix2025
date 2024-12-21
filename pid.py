class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0):
        """
        Initialize the PID controller.

        :param kp: Proportional gain
        :param ki: Integral gain
        :param kd: Derivative gain
        :param setpoint: Desired setpoint
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint

        self.previous_error = 0
        self.integral = 0

    def update(self, current_value, dt):
        """
        Calculate the control variable using the PID algorithm.

        :param current_value: Current value of the process variable
        :param dt: Time interval since the last update
        :return: Control variable
        """
        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt

        output = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )

        self.previous_error = error
        return output

# Example usage:
if __name__ == "__main__":
    import time

    # Initialize the PID controller with desired gains
    pid = PIDController(kp=1.0, ki=0.1, kd=0.05, setpoint=100)

    # Simulate a process variable starting at 50
    process_variable = 50
    dt = 0.1  # 100 ms time step

    for _ in range(500):
        control = pid.update(process_variable, dt)
        process_variable += control * dt  # Simulate process reaction

        print(f"Process Variable: {process_variable:.2f}, Control: {control:.2f}")
        time.sleep(dt)
