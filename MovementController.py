import math

import numpy as np
from scipy.spatial.distance import cdist
from itertools import permutations
import time

class Robot:
    def __init__(self, x, y, rotation):
        self.x = x
        self.y = y
        self.rotation = rotation


def next_command_from_state(state):
    robot_positions = robot_front_and_back(state.robot)
    if(robot_positions is None):
        return ""


    robot_pos = robot_position(robot_positions)

    print("Robot rotation: ", robot_rotation(robot_positions))

    robot = Robot(robot_pos[0], robot_pos[0], robot_rotation(robot_positions))
    ball_vectors = vectors_to_balls(robot, state.balls)
    vector = [0, 0]
    if len(ball_vectors) == 1:
        vector = ball_vectors[0]
    elif len(ball_vectors) > 1:
        vector = shortest_vector(ball_vectors)
    if vector[0] == 0 and vector[1] == 0:
        return "cMove 0"  # Stops the movement because we didn't find any balls TODO: Make this look for the goal'

    aim_rotation = angle_of_vector(vector[0], vector[1])  # Checking for rotation

    if aim_rotation - robot.rotation > 5:
        print("robot rotation: ", robot.rotation)
        print("Should be rotation:", aim_rotation)

        command = f"turn {int(aim_rotation - robot.rotation)}"
        if aim_rotation - robot.rotation > 180:
            command = f"turn {360 - int(aim_rotation - robot.rotation)}"
            print("rotating by: ", 360 - aim_rotation - robot.rotation)
        else:
            print("rotating by: ", aim_rotation - robot.rotation)
        print(command)
        return command
    else:
        return "move 30"


def robot_rotation(position):
    return angle_of_vector(position[0][0]-position[1][0], position[0][1] - position[1][1])


def robot_position(front_and_back):
    diff = [front_and_back[0][0] - front_and_back[1][0], front_and_back[0][1] - front_and_back[1][1]]
    return [front_and_back[0][0] - (diff[0] / 2), front_and_back[0][1] - (diff[1] / 2)]


def robot_front_and_back(robot_state):

    if robot_state is None or robot_state.pos_1 is None or robot_state.pos_2 is None or robot_state.pos_3 is None:
        print("Robot not found!")
        return None

    positions = [robot_state.pos_1, robot_state.pos_2, robot_state.pos_3]

    print("Positions: ", positions)

    # Positions: [[x, y],[x, y], [x, y]]
    # Three positions determines location.
    # The two closest positions are the two front positions.
    # The last is the back.
    # By finding the spot between the back positions and using the front position, determine the rotation and position.

    vectors_between = [
        [positions[0][0] - positions[1][0], positions[0][1] - positions[1][1]],
        [positions[1][0] - positions[2][0], positions[1][1] - positions[2][1]],
        [positions[2][0] - positions[0][0], positions[2][1] - positions[0][1]],
    ]
    shortest = shortest_vector(vectors_between)
    front = []
    back = []
    if shortest == vectors_between[0]:
        front = [positions[0][0] - shortest[0] / 2, positions[0][1] - shortest[1] / 2]
        back = positions[2]
    elif shortest == vectors_between[1]:
        front = [positions[1][0] - shortest[0] / 2, positions[1][1] - shortest[1] / 2]
        back = positions[0]
    elif shortest == vectors_between[2]:
        front = [positions[2][0] - shortest[0] / 2, positions[2][1] - shortest[1] / 2]
        back = positions[1]
    
    print([front, back])
    return [front, back]


def angle_of_vector(x, y):
    angle = math.degrees(math.atan2(y, x))
    if angle < 0:
        angle += 360
    # Invert the angle
    angle = 360 - angle
    # Ensure the angle is within 0-360 degrees
    if angle >= 360:
        angle -= 360
    return angle


def vectors_to_balls(robot, ball_states):
    ball_vectors = []
    for ball in ball_states:
        ball_vectors.append(vector_from_robot_to_next_ball(robot, ball))
    return ball_vectors


def shortest_vector(vectors):
    shortest = vectors[0]
    for vector in vectors:
        if vector_length(vector) < vector_length(shortest):
            shortest = vector
    return shortest


def vector_length(vector):
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2)


def vector_from_robot_to_next_ball(robot, ball):
    return np.array([robot.x - ball.x, robot.y - ball.y])




def extract_ball_positions(ball_objects):
    positions = [(ball.x, ball.y) for ball in ball_objects]

    return np.array(positions)


## Bruteforce to find best route to pick up balls
def find_shortest_path(robot_position, ball_positions):
    
    start_time = time.time()


    ball_positions = extract_ball_positions(ball_positions)

    points = np.vstack([[0, 0], ball_positions])

    # pariwise distancematrix
    dist_matrix = cdist(points, points)

    # Generate all combinations of ball indices
    num_balls = len(ball_positions)
    ball_indices = range(1, num_balls + 1) 
    all_routes = permutations(ball_indices)

    # Total distance function
    def calculate_total_distance(route):
        total_dist = 0
        current_pos = 0 
        for next_pos in route:
            total_dist += dist_matrix[current_pos, next_pos]
            current_pos = next_pos
        return total_dist

    # Find the route with the minimum distance
    min_distance = float('inf')
    best_route = None
    for route in all_routes:
        distance = calculate_total_distance(route)
        if distance < min_distance:
            min_distance = distance
            best_route = route

    
    best_route_positions = [points[0]] + [points[idx] for idx in best_route]

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time to find the shortest path: {elapsed_time:.2f} seconds")

    return best_route_positions, min_distance

def nearest_neighbor_path(robot, ball_objects):
    start_time = time.time() 
    
    robot_pos = [0, 0]

    ball_positions = extract_ball_positions(ball_objects)

    points = np.vstack([robot_pos, ball_positions])

    # pairwise distance matrix
    dist_matrix = cdist(points, points)
    
    num_points = len(points)
    visited = np.zeros(num_points, dtype=bool)
    visited[0] = True 
    path = [0]  

    # Greedy Nearest Neighbor Algorithm
    current_index = 0
    while len(path) < num_points:
        distances = dist_matrix[current_index]
        print("distances: ",distances)
        distances[visited] = np.inf  
        next_index = np.argmin(distances)  # Find the index of the nearest unvisited point
        path.append(next_index) 
        visited[next_index] = True 
        current_index = next_index 

    
    path_positions = [points[idx] for idx in path[1:]]
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time to find the path: {elapsed_time:.2f} seconds")
    print("PATH: ", path_positions)

    return path_positions