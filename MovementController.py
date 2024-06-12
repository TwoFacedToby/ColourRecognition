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

class Ball:
    def __init__(self, x, y, is_orange):
        self.x = x
        self.y = y
        self.isOrange = is_orange


# Global variable to store the target ball position
current_target_ball = None

def ball_is_present(target_ball, ball_positions, error_margin=5):
    """ Check if the target ball is still present within an error margin. """
    for ball in ball_positions:
        if np.linalg.norm(np.array([target_ball.x, target_ball.y]) - np.array([ball.x, ball.y])) < error_margin:
            return True
    return False

def next_command_from_state(state):
    global current_target_ball

    robot_positions = robot_front_and_back(state.robot)
    if robot_positions is None:
        return "", None  # Return None for coordinates if no robot is found

    robot_pos = robot_position(robot_positions)
    robot = Robot(robot_pos[0], robot_pos[1], robot_rotation(robot_positions))

    ball_positions = state.balls  # Corrected usage

    # Debug print to check the ball positions
    print("ball_positions:", ball_positions)

    ball_vectors = vectors_to_balls(robot, state.balls)
    vector = [0, 0]
    closest_ball_coords = None

    # Check if the current target ball is still present
    if current_target_ball and ball_is_present(current_target_ball, ball_positions):
        vector = vector_from_robot_to_next_ball(robot, current_target_ball)
        closest_ball_coords = (current_target_ball.x, current_target_ball.y)
        print("Still targeting the same ball")
    else:
        current_target_ball = None  # Reset if the ball is no longer present

    if current_target_ball is None:
        print("Finding new ball!")
        if len(ball_vectors) == 1:
            vector = ball_vectors[0]
            closest_ball_coords = (ball_positions[0].x, ball_positions[0].y)
            current_target_ball = ball_positions[0]
        elif len(ball_vectors) > 1:
            vector, closest_ball_index = shortest_vector_with_index(ball_vectors)
            closest_ball_coords = (ball_positions[closest_ball_index].x, ball_positions[closest_ball_index].y)
            current_target_ball = ball_positions[closest_ball_index]
        elif not ball_positions:  # No balls left
            print("Navigating to goal!")
            return navigate_to_goal(robot, state.big_goal_pos)


    print("Rotation: ", robot_rotation(robot_positions))

    if vector[0] == 0 and vector[1] == 0:
        return "cMove 0", None, None  # Return None if no balls are found

    aim_rotation = angle_of_vector_t(-vector[0], -vector[1])  # Checking for rotation

    print("Vector: ", vector[0], " ", vector[1])
    print("Angle: ", aim_rotation)

    temp = normalize_angle_difference(robot_rotation(robot_positions), aim_rotation)
    print("To rotate: ", temp)

    distance = vector_length(vector)

    if -1 < temp < 1:
        return f"move {int(np.abs(distance * 1.7))}"
    else:
        return f"turn {int(temp)}"
    
def calculate_distance(point1, point2):
    """ Calculate the Euclidean distance between two points. """
    return np.linalg.norm(np.array(point1) - np.array(point2))

def navigate_to_goal(robot, goal_position):
    if not goal_position:
        return "cMove 0"  # No goal position available

    robot_x, robot_y = robot.x, robot.y
    goal_x, goal_y = goal_position

    

    # Calculate the distance to the goal
    distance_to_goal = calculate_distance((robot_x, robot_y), goal_position)


    print("Distance to goal: ", distance_to_goal)

    if distance_to_goal <= 110:
        print("Robot is at the goal.")
        print()
        return "brush 80"

    # Calculate the vectors for up/down and left/right movements
    vertical_vector = np.array([0, goal_y - robot_y])
    horizontal_vector = np.array([goal_x - robot_x, 0])

    print("vv", vertical_vector[1])
    if abs(vertical_vector[1]) > 30:  # Move vertically first if significant distance
        aim_rotation = angle_of_vector_t(vertical_vector[0], vertical_vector[1])
        distance = vector_length(vertical_vector)
        temp = normalize_angle_difference(robot.rotation, aim_rotation)

        if -1.4 < temp < 1.4:
            print(f"veri move {int(np.abs(distance * 1.4))}")
            return f"move {int(np.abs(distance * 1.4))}"
        else:
            print(f"veri turn {int(temp)}")
            return f"turn {int(temp)}"
    elif abs(horizontal_vector[0]) > 10:  # Then move horizontally if significant distance
        aim_rotation = angle_of_vector_t(horizontal_vector[0], horizontal_vector[1])
        distance = vector_length(horizontal_vector)
        temp = normalize_angle_difference(robot.rotation, aim_rotation)

        if -1.4 < temp < 1.4:
            print(f"hori move {int(np.abs(distance * 1.4))}")
            return f"move {int(np.abs(distance * 1.4))}"
        else:
            print(f"hori turn {int(temp)}")
            return f"turn {int(temp)}"
    
    return "cMove 0"  # If already at goal

def get_all_ball_positions(ball_states):
    """
    Extracts and returns the positions of all balls as a list of tuples (x, y).
    """
    # Debug print to check the ball states being passed
    print("ball_states:", ball_states)
    
    ball_positions = []
    for ball in ball_states:
        # Print each ball's attributes to understand the format
        print("ball:", ball, "ball.x:", ball.x, "ball.y:", ball.y)
        ball_positions.append((ball.x, ball.y))
    
    # Debug print to check the extracted ball positions
    print("ball_positions:", ball_positions)
    
    return ball_positions

def normalize_angle_difference(angle1, angle2):
    difference = (angle1 - angle2 + 180) % 360 - 180
    return difference if difference != -180 else 180


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
    shortest, index = shortest_vector_with_index(vectors_between)
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

def angle_of_vector_t(x, y):
    angle = math.degrees(math.atan2(y, x))
    if angle < 0:
        angle += 360
    # Invert the angle to correct the mirroring issue
    angle = (360 - angle) % 360
    return angle



def vectors_to_balls(robot, ball_states):
    ball_vectors = []
    for ball in ball_states:
        ball_vectors.append(vector_from_robot_to_next_ball(robot, ball))
    return ball_vectors


def shortest_vector_with_index(vectors):
    shortest = vectors[0]
    shortest_index = 0
    for i, vector in enumerate(vectors):
        if vector_length(vector) < vector_length(shortest):
            shortest = vector
            shortest_index = i
    return shortest, shortest_index


def vector_length(vector):
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2)


def vector_from_robot_to_next_ball(robot, ball):
    return np.array([robot.x - ball.x, robot.y - ball.y])




def extract_ball_positions(ball_objects):
    positions = [(ball.x, ball.y) for ball in ball_objects]

    return np.array(positions)

'''New Adventure'''
def line_intersect_detecter(line, wall):

    for wall in walls:
        wx1, wy1, wx2, wy2 = wall

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        # Check if any segment of the line intersects with any segment of the wall rectangle
        if intersect((line[0], line[1]), (line[2], line[3]), (wx1, wy1), (wx2, wy1)) \
                or intersect((line[0], line[1]), (line[2], line[3]), (wx2, wy1), (wx2, wy2)) \
                or intersect((line[0], line[1]), (line[2], line[3]), (wx2, wy2), (wx1, wy2)) \
                or intersect((line[0], line[1]), (line[2], line[3]), (wx1, wy2), (wx1, wy1)):
            return True

    return False


def find_path_around_wall()
    #Write me

## Bruteforce to find best route to pick up balls
def find_shortest_path(robot_position, ball_positions, wall_positions):
    
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

def nearest_neighbor_path(robot, ball_objects, walls):
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

    #Check of the path intersect a WALL
        line = (points[current_index][0], points[current_index][1], points[next_index][0], points[next_index][1])
        for wall in walls:
            if line_intersect_detecter(line, wall):
            #New PATH!

            break #No need to check more walls as we have a new path


        path.append(next_index)
        visited[next_index] = True 
        current_index = next_index 

    
    path_positions = [points[idx] for idx in path[1:]]
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time to find the path: {elapsed_time:.2f} seconds")
    print("PATH: ", path_positions)

    return path_positions