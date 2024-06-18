import math

import numpy as np
from scipy.spatial.distance import cdist
from itertools import permutations
import time
import shared_state
import heapq


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




movementSpeed = 30
turnSpeed = 20
wheel_diameter = 70  # mm
wheel_circumference = math.pi * wheel_diameter
wheel_distance = 170  # mm
robot_circumference = math.pi * wheel_distance



def ball_is_present(target_ball, ball_positions, error_margin=5):
    """ Check if the target ball is still present within an error margin. """
    for ball in ball_positions:
        if np.linalg.norm(np.array([target_ball.x, target_ball.y]) - np.array([ball.x, ball.y])) < error_margin:
            return True
    return False

# Given values
robot_real_height = 16.0  # cm
camera_height = 189  # cm

def calculate_real_world_position(robot_pos, image_height):
    """
    Calculate the real-world position of the robot.
    
    Parameters:
    robot_pos (tuple): The observed position of the robot in the image (x, y).
    image_height (int): The height of the image in pixels.
    
    Returns:
    tuple: The real-world position of the robot (x, y) in pixels for drawing.
    """
    # Calculate the vertical distance from the robot's top to the camera height
    vertical_distance = camera_height - robot_real_height

    # Calculate the real-world vertical position based on camera height
    # This involves projecting the observed position to the ground
    # Use the ratio of the camera height to the vertical distance
    real_world_y = robot_pos[1] + (robot_real_height / camera_height) * (image_height - robot_pos[1])

    # The x-coordinate is unaffected in this simplified top-down projection
    real_world_x = robot_pos[0]

    return real_world_x, real_world_y

def robot_rotation_old(position):
    return angle_of_vector(position[0][0]-position[1][0], position[0][1] - position[1][1])

'''Newly added wall avoidance functions starts here. Most of them need point, the grid and how big each grid cell is.'''

'''Checks if a path is clear, if not it returns the grid with an obstacle in the way'''
def is_path_clear(grid, start, end, cell_height, cell_width):
    x0, y0 = start

    x1 = end.x
    y1 = end.y

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x = x0
    y = y0
    n = 1 + dx + dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    error = dx - dy

    dx *= 2
    dy *= 2

    for _ in range(int(n)):
        grid_x = x // cell_width
        grid_y = y // cell_height
        if grid[int(grid_y)][int(grid_x)] == 1:
            #return (grid_y, grid_x)  # Obstacle found at this grid
            return False
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

    return True # Safe to proceed

'''Two helper functions for the algortihm of avoiding balls'''
def heuristic(a, b): #Defines how good a move is
    # Using Manhattan distance as heuristic
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def is_valid_move(grid, current, neighbor): #Checks we dont go through any walls, if so not valid move in algorithm
    rows, cols = len(grid), len(grid[0])
    if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
        return False
    if grid[neighbor[0]][neighbor[1]] == 1:
        return False
    if abs(current[0] - neighbor[0]) == 1 and abs(current[1] - neighbor[1]) == 1: # Diagonal movement
        if grid[current[0]][neighbor[1]] == 1 or grid[neighbor[0]][current[1]] == 1:
            return False
    return True

'''This function creates a path using A* algorithm to a goal around walls.
It does not yet take the width of the robot into acount and can "only" move in 8 directions.
The steps are going through each grid so we translate it into an end coordinate with another function
It takes some energy and should only be run if there is found an obstacle in the direct path'''
def path_around_wall(grid, start, end):
    rows, cols = len(grid), len(grid[0])
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            if is_valid_move(grid, current, neighbor):
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found



# Define global variables to keep track of the state and vectors
global_step = 0
V_parallel = None
V_perpendicular = None
vectors_initialized = False


'''This is a helper function it turns the path from path_around_wall and turns it into
an end coordinate, specificly the last grid spot before the robot needs to turn.'''
def find_next_step_passt_wall(path, cell_height, cell_width):
    if len(path) < 2:
        return (path[0][0] * cell_height, path[0][1] * cell_width)  # There is only one way to go

    # Initialize the direction vector
    direction = (path[1][0] - path[0][0], path[1][1] - path[0][1])

    for i in range(1, len(path) - 1):
        current_direction = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if current_direction != direction:
            return (path[i][0] * cell_height, path[i][1] * cell_width)  # Return coordinates at strightline end

    return (path[-1][0] * cell_height, path[-1][1] * cell_width)  # The whole path is straight

'''End of new functions for wall avoidance'''

def next_command_from_state(state):
    global current_target_ball
    global global_step

    robot_positions = robot_front_and_back(state.robot)
    if robot_positions is None:
        return "", None  # Return None for coordinates if no robot is found

    robot_pos = robot_position(robot_positions)
    
    # Calculate the real-world position of the robot
    real_robo_pos = shared_state.real_position_robo
    robot = Robot(real_robo_pos[0], real_robo_pos[1], robot_rotation_old(robot_positions))

    ball_positions = state.balls  # Corrected usage

    

    ball_vectors = vectors_to_balls(robot, ball_positions)
    vector = [0, 0]
    closest_ball_coords = None

    # Check if the current target ball is still present
    if current_target_ball and ball_is_present(current_target_ball, ball_positions):
        vector = vector_from_robot_to_next_ball(robot, current_target_ball)
        closest_ball_coords = (current_target_ball.x, current_target_ball.y)
        print("Still targeting the same ball")
    elif global_step == 2:
        print("Dont wait for backing")
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
        

    
    shared_state.current_ball = current_target_ball

    


    
    



    '''
    if not is_path_clear(shared_state.current_grid, real_robo_pos, current_target_ball, shared_state.current_cell_height, shared_state.current_cell_width):
        temp = (current_target_ball.x, current_target_ball.y)
        path = path_around_wall(shared_state.current_grid, real_robo_pos, temp)
        next_coord = find_next_step_passt_wall(path, shared_state.current_cell_height, shared_state.current_cell_width)
        print("There is an obstalce. This is the next coord: ", next_coord)
    '''

    wall_prox = check_wall_proximity(current_target_ball.x, current_target_ball.y)

    
    print("Wall prox: ", wall_prox)
    


    if wall_prox or global_step == 2:
        global V_parallel, V_perpendicular, vectors_initialized

        if not vectors_initialized:
            closest_wall, V_parallel, V_perpendicular = handle_ball_near_wall(current_target_ball.x, current_target_ball.y, vector)
            vectors_initialized = True

        if global_step == 0:
            # Perform the initial action for step 0
            vector = V_parallel
            aim_rotation = angle_of_vector_t(-vector[0], -vector[1])  # Checking for rotation
            temp = normalize_angle_difference(robot.rotation, aim_rotation)

            print("Vector: ", -vector[0], -vector[1])
            print("Rot", aim_rotation)

            # Calculate the distance and normalize it using the reference vector magnitude and real world distance
            distance = vector_length(vector)
            normalized_distance = (840/shared_state.half_field_pixel) * distance

            if -1 < temp < 1:
                print("step 1 action: forward")
                global_step = 1
                return f"forward_degrees {int(forward(normalized_distance - 40))} {movementSpeed}"
            else:
                print("step 1 action: turn")
                return f"turn_degrees {int(turn(temp * 2))} {turnSpeed}"

        elif global_step == 1:
            aim_rotation = angle_of_vector_t(-vector[0], -vector[1])  # Checking for rotation
            temp = normalize_angle_difference(robot.rotation, aim_rotation)

            # Calculate the distance and normalize it using the reference vector magnitude and real world distance
            distance = vector_length(vector)
            normalized_distance = (840/shared_state.half_field_pixel) * distance

            if -1 < temp < 1:
                print("step 2 action: forward")
                global_step = 2
                return f"forward_degrees {int(forward(normalized_distance-150))} {movementSpeed}"
            else:
                print("step 2 action: turn")
                return f"turn_degrees {int(turn(temp * 2))} {turnSpeed}"

        elif global_step == 2:
            print("HALLO")
            vector = V_perpendicular

            # Calculate the distance and normalize it using the reference vector magnitude and real world distance
            distance = vector_length(vector)
            normalized_distance = (840/shared_state.half_field_pixel) * -distance

            distance_tweak = normalized_distance + 250
            
            print("step 3 action: backward")
            print("DISTANCE: ", normalized_distance)
            global_step = 0  # Reset the state
            vectors_initialized = False  # Reset the initialization flag
            return f"forward_degrees {int(forward(distance_tweak))} {movementSpeed}"
            



    aim_rotation = angle_of_vector_t(-vector[0], -vector[1])  # Checking for rotation

    temp = normalize_angle_difference(robot.rotation, aim_rotation)
    

    # Calculate the distance and normalize it using the reference vector magnitude and real world distance
    distance = vector_length(vector)

    #print("Distance between ball and robot: ", distance, " and reference vector ", shared_state.reference_vector_magnitude)
    normalized_distance = (840/shared_state.half_field_pixel) * distance 
    #print("Normalized distance: ", normalized_distance)

    if -1 < temp < 1:
        return f"forward_degrees {int(forward(normalized_distance-200))} {movementSpeed}"
    else:
        return f"turn_degrees {int(turn(temp*2))} {turnSpeed}"



def decompose_vector(vector, wall_orientation):
    """
    Decomposes the given vector into two perpendicular components 
    based on the wall orientation.

    Parameters:
    - vector: A tuple (Vx, Vy) representing the vector.
    - wall_orientation: 'vertical' or 'horizontal' indicating the orientation of the wall.

    Returns:
    - (V_parallel, V_perpendicular): Two vectors representing the components parallel and perpendicular to the wall.
    """
    Vx, Vy = vector
    
    if wall_orientation == 'vertical':
        # Wall is vertical, so parallel component is horizontal
        V_parallel = (0, Vy)
        V_perpendicular = (Vx, 0)
    elif wall_orientation == 'horizontal':
        # Wall is horizontal, so parallel component is vertical
        V_parallel = (Vx, 0)
        V_perpendicular = (0, Vy)
    else:
        raise ValueError("wall_orientation must be 'vertical' or 'horizontal'")
    
    return V_parallel, V_perpendicular

def check_wall_proximity(ball_x, ball_y, threshold=40):
    """
    Checks which wall the ball is closest to within a given threshold.
    
    Parameters:
    - ball_x: x-coordinate of the ball.
    - ball_y: y-coordinate of the ball.
    - left_wall_x: x-coordinate of the left wall.
    - right_wall_x: x-coordinate of the right wall.
    - top_wall_y: y-coordinate of the top wall.
    - bottom_wall_y: y-coordinate of the bottom wall.
    - threshold: Distance threshold to consider the ball close to the wall (default is 40 pixels).
    
    Returns:
    - A string indicating which wall the ball is closest to, or None if it's not close to any wall.
    """
    if abs(ball_x - shared_state.left_wall) <= threshold:
        print("left")
        return 'left'
    elif abs(ball_x - shared_state.right_wall) <= threshold:
        print("right")
        return 'right'
    elif abs(ball_y - shared_state.upper_wall) <= threshold:
        print("top")
        return 'top'
    elif abs(ball_y - shared_state.lower_wall) <= threshold:
        print("bottom")
        return 'bottom'
    else:
        return None

def handle_ball_near_wall(ball_x, ball_y, vector, threshold=40):
    """
    Determines the closest wall and decomposes the ball's movement vector accordingly.
    
    Parameters:
    - ball_x: x-coordinate of the ball.
    - ball_y: y-coordinate of the ball.
    - vector: A tuple (Vx, Vy) representing the ball's current movement vector.
    - left_wall_x: x-coordinate of the left wall.
    - right_wall_x: x-coordinate of the right wall.
    - top_wall_y: y-coordinate of the top wall.
    - bottom_wall_y: y-coordinate of the bottom wall.
    - threshold: Distance threshold to consider the ball close to the wall (default is 40 pixels).
    
    Returns:
    - A tuple containing the closest wall and the decomposed vectors (V_parallel, V_perpendicular).
    """
    closest_wall = check_wall_proximity(ball_x, ball_y, threshold)
    
    if closest_wall:
        if closest_wall in ['left', 'right']:
            wall_orientation = 'vertical'
        elif closest_wall in ['top', 'bottom']:
            wall_orientation = 'horizontal'
        
        V_parallel, V_perpendicular = decompose_vector(vector, wall_orientation)
        return closest_wall, V_parallel, V_perpendicular
    else:
        return None, None, None




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

        if -1 < temp < 1:
            return f"forward_degrees {int(forward(np.abs(distance*1.4)))} {movementSpeed}"
        else:
            print(f"veri turn {int(temp)}")
            return f"turn_degrees {int(turn(temp*2))} {turnSpeed}"
        
    elif abs(horizontal_vector[0]) > 10:  # Then move horizontally if significant distance
        aim_rotation = angle_of_vector_t(horizontal_vector[0], horizontal_vector[1])
        distance = vector_length(horizontal_vector)
        temp = normalize_angle_difference(robot.rotation, aim_rotation)

        if -1 < temp < 1:
            print(f"hori move {int(np.abs(distance * 1.4))}")
            return f"forward_degrees {int(forward(np.abs(distance*1.4)))} {movementSpeed}"
        else:
            print(f"hori turn {int(temp)}")
            return f"turn_degrees {int(turn(temp*2))} {turnSpeed}"
    
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


def robot_rotation(position, image_height):
    # Convert the front and back positions to real-world coordinates
    front_real_world = calculate_real_world_position(position[0], image_height)
    back_real_world = calculate_real_world_position(position[1], image_height)
    return angle_of_vector(front_real_world[0] - back_real_world[0], front_real_world[1] - back_real_world[1])


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

def turn (angle):
    angle = angle*-0.45
    turn_circumference = (angle / (360)) * robot_circumference
    rotations = turn_circumference / wheel_circumference
    return rotations * 360


def forward(distance):
        rotations = distance / wheel_circumference
        return rotations * 360