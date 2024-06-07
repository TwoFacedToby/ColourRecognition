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
    robot = robot_location(state.robot)
    ball_vectors = vectors_to_balls(robot, state.balls)
    vector = [0, 0]
    if len(ball_vectors) == 1:
        vector = ball_vectors[0]
    elif len(ball_vectors) > 1:
        vector = shortest_vector(ball_vectors)
    if vector[0] == 0 and vector[1] == 0:
        return "cMove 0"  # Stops the movement because we didn't find any balls TODO: Make this look for the goal'

    aim_rotation = angle_of_vector(vector[0], vector[1])  # Checking for rotation
    print("Should be rotation:", aim_rotation)
    if aim_rotation > robot.rotation:
        if aim_rotation > 180 + robot.rotation:
            return "cTurn -20"  # turn counter-clockwise
        else:
            return "cTurn 20"  # turn clockwise
    if aim_rotation < robot.rotation:
        if aim_rotation < robot.rotation-180:
            return "cTurn 20"  # turn clockwise
        else:
            return "cTurn -20"  # turn counter-clockwise

    return ""


def robot_location(robot_state):
    positions = [robot_state.pos_1, robot_state.pos_2, robot_state.pos_3]
    # Positions: [[x, y],[x, y], [x, y]]
    # Three positions determines location.
    # The two closest positions are the two back positions.
    # The last is the front.
    # By finding the spot between the back positions and using the front position, determine the rotation and position.

    return Robot(0, 0, 0)


def angle_of_vector(x, y):
    return math.degrees(math.atan2(-y, x))


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
    print("ball x: ", ball.x, " ball y: ", ball.y)
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