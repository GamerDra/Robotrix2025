import neat
import numpy as np
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import sys
import traceback
import random

# Initialize the connection with CoppeliaSim
client = RemoteAPIClient()
sim = client.getObject('sim')  # Reference simulation API
sensor = sim.getObject('/basket_bot/hoop_sensor')

# Hoop and ball handlers
hoop = sim.getObjectHandle('/basket_bot/rail_1')
rail_2 = sim.getObjectHandle('/basket_bot/rail_2')
ball = sim.getObjectHandle('ball')

# Global variables for scoring
score = 0
prev_score = 0

# Function to check if the ball passed through the hoop
def check_ball_through_hoop():
    global score, prev_score
    det = sim.readProximitySensor(sensor)
    # print(f"Sensor detection: {det}")
    if det[0] == 1 and prev_score != det[0]:
        print("Scored!")
        score += 1
        prev_score = det[0]
        return True
    prev_score = det[0]
    return False

# Function to reset the hoop's position
def reset_hoop_position():
    print("Resetting hoop to default position.")
    sim.setObjectPosition(hoop, [0, -3.7, 0.230])
    sim.setObjectPosition(rail_2, [0.360, -3.70056, 0.2305])

# Function to get the ball's position
def get_ball_position():
    ball_pos = sim.getObjectPosition(ball)
    # print(f"Ball position: {ball_pos}")   
    return ball_pos

# Function to set velocities for the hoop actuators
def set_velocity(x, y, z):
    max_velocity = 2.0  # Define a suitable range for velocity
    scaled_x = x * max_velocity
    scaled_y = y * max_velocity
    scaled_z = z * max_velocity
    # print(f"Setting velocity to ({scaled_x}, {scaled_y}, {scaled_z})")
    joint_x_obj = sim.getObject('/basket_bot/actuator_x')
    joint_y_obj = sim.getObject('/basket_bot/actuator_y')
    joint_z_obj = sim.getObject('/basket_bot/actuator_z')
    sim.setJointTargetVelocity(joint_x_obj, -scaled_x)
    sim.setJointTargetVelocity(joint_y_obj, scaled_y)
    sim.setJointTargetVelocity(joint_z_obj, -scaled_z)

# Function to launch balls with randomized trajectories
def launch_ball(force):
    z = random.uniform(190,230)
    y = (force / z) + random.uniform(0, 30)
    x = random.uniform(-40, 40)  # Horizontal force component (X-axis)
    
    # Reset the ball's position and orientation
    sim.setObjectPosition(ball, [0, 4.5, 0.605])
    sim.setObjectOrientation(ball, [0, 0, 0])
    
    # Apply force to the ball
    sim.addForceAndTorque(ball, [x, -y, z], [0, 0, 0])  # No torque applied
    # print(f"Launched ball with force: x={x}, y={-y}, z={z}")

# Fitness function for NEAT
def evaluate_genomes(genomes, config):
    for genome_id, genome in genomes:
        reset_hoop_position()
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        fitness = 0
        num_balls = 5  # Number of balls to launch
        
        for i in range(num_balls):
            launch_ball(force = 11000)  # Launch a ball with base force 100
            
            for step in range(100):  # Simulation steps for each ball
                ball_pos = get_ball_position()
                inputs = [ball_pos[0], ball_pos[1], ball_pos[2]]
                outputs = net.activate(inputs)

                # Control hoop movement
                set_velocity(outputs[0], outputs[1], outputs[2])

                # Check if ball goes through the hoop
                if check_ball_through_hoop():
                    print(f"Ball {i} scored!")
                    fitness += 1000  # Reward for scoring
                
                # Penalize distance between ball and hoop
                hoop_pos = sim.getObjectPosition(hoop, -1)
                distance = np.linalg.norm(np.array(ball_pos) - np.array(hoop_pos))
                fitness -= distance / 100  # Penalize larger distances

        genome.fitness = fitness

# NEAT configuration and training
def run_neat():
    config_path = "config-feedforward"  # Path to your NEAT configuration file
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Initialize the population
    population = neat.Population(config)

    # Add reporters to show progress
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Run NEAT algorithm
    winner = population.run(evaluate_genomes, n=50)  # Number of generations

    # Save the best genome
    with open("winner.pkl", "wb") as f:
        import pickle
        pickle.dump(winner, f)

# Main function to start the simulation
if __name__ == "__main__":
    client = RemoteAPIClient()
    sim = client.getObject('sim')

    try:
        # Start the simulation
        try:
            return_code = sim.startSimulation()
            if sim.getSimulationState() != sim.simulation_stopped:
                print('\nSimulation started correctly in CoppeliaSim.')
            else:
                print('\nSimulation could not be started correctly in CoppeliaSim.')
                sys.exit()
        except Exception:
            print('\n[ERROR] Simulation could not be started!')
            traceback.print_exc(file=sys.stdout)
            sys.exit()

        # Run the NEAT control logic
        try:
            run_neat()
        except Exception:
            print('\n[ERROR] Your control_logic function threw an Exception, kindly debug your code!')
            print('Stop the CoppeliaSim simulation manually if required.\n')
            traceback.print_exc(file=sys.stdout)
            print()
            sys.exit()

        # Stop the simulation
        try:
            return_code = sim.stopSimulation()
            time.sleep(0.5)
            if sim.getSimulationState() == sim.simulation_stopped:
                print('\nSimulation stopped correctly in CoppeliaSim.')
            else:
                print('\nSimulation could not be stopped correctly in CoppeliaSim.')
                sys.exit()
        except Exception:
            print('\n[ERROR] Simulation could not be stopped!')
            traceback.print_exc(file=sys.stdout)
            sys.exit()
    except KeyboardInterrupt:
        # Stop the simulation in case of interruption
        return_code = sim.stopSimulation()
        time.sleep(0.5)
        if sim.getSimulationState() == sim.simulation_stopped:
            print('\nSimulation interrupted by user in CoppeliaSim.')
        else:
            print('\nSimulation could not be interrupted. Stop the simulation manually.')
            sys.exit()
