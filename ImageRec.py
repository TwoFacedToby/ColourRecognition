import numpy as np
import cv2
from enum import Enum
from scipy.stats import mode
from collections import Counter
from scipy.stats import circmean
from MovementController import next_command_from_state, robot_position, robot_front_and_back, shortest_vector_with_index, vector_from_robot_to_next_ball
# Capturing video through webcam
cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
from sklearn.cluster import DBSCAN
from collections import Counter, deque
import shared_state
import math


#cam = cv2.VideoCapture("TrackVideos/Tester.mp4")

class State:
    def __init__(self, balls, corners, robot, small_goal_pos, big_goal_pos):
        self.balls = balls
        self.corners = corners
        self.robot = robot
        self.small_goal_pos = small_goal_pos
        self.big_goal_pos = big_goal_pos


class Ball:
    def __init__(self, x, y, is_orange):
        self.x = x
        self.y = y
        self.isOrange = is_orange


class Robot:
    def __init__(self, pos_1, pos_2, pos_3):
        self.pos_1 = pos_1
        self.pos_2 = pos_2
        self.pos_3 = pos_3


class Type(Enum):
    Robot = [255, 255, 0]
    Wall = [0, 0, 0]
    Ball = [200, 200, 0]
    OrangeBall = [200, 255, 0]
    SmallBlueGoal = [255, 101, 0]
    BigGreenGoal = [30, 170, 70]
    Corner = [229, 178, 23]


class Color(Enum):
    WHITE = 0
    ORANGE = 1
    RED = 2
    BROWN = 3
    GREEN = 4
    BLUE = 5
    YELLOW = 6
    BLACK = 7


ballsState = []
wallsState = []
smallGoalPos = []
bigGoalPos = []
white_egg = []
robot_state = Robot([0, 0], [0, 0], [0, 0])
walls = []
temp_lines = []
all_lines = []




def recognise_state_and_draw(image, mask, types):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    color = types.value
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        show = "none"

        match types:
            case Type.Wall:
                if 10 < area < 300:
                    show = "square"

            case Type.SmallBlueGoal:
                if 100 < area < 1500:
                    smallGoalPos.clear()
                    smallGoalPos.append((x, y))
                    show = "square"

            case Type.BigGreenGoal:
                if 20 < area < 1500:
                    bigGoalPos.clear()
                    bigGoalPos.append([x, y])
                    show = "square"

            case Type.Ball:
                if 120 < area < 500:
                    ballsState.append(Ball(x=x, y=y, is_orange=False))
                    show = "circle"
                elif 500 < area < 2500:
                    global white_egg
                    white_egg = [x, y]
                    show = "square"

            case Type.OrangeBall:
                if 120 < area < 1500:
                    ballsState.append(Ball(x=x, y=y, is_orange=True))
                    show = "circle"

            case Type.Robot:
                if 400 < area < 1500 and h/2 < w < h*2 and w/2 < h < w*2:
                    if robot_state.pos_1 == [0, 0]:
                        robot_state.pos_1 = [x, y]
                    elif robot_state.pos_2 == [0, 0]:
                        robot_state.pos_2 = [x, y]
                    elif robot_state.pos_3 == [0, 0]:
                        robot_state.pos_3 = [x, y]
                        print(robot_state.pos_1, " ", robot_state.pos_2, " ", robot_state.pos_3)

                    show = "square"
        if show == "circle":
            cv2.circle(image, (x + 8, y + 8), 8, (color[0], color[1], color[2]), 2)
        elif show == "square":
            cv2.rectangle(image, (x, y), (x + w, y + h), (color[0], color[1], color[2]), 2)


def mask_from_color(color, image_hsv, clean_image):
    color_low = []
    color_high = []
    match color:
        case Color.WHITE:
            color_low = np.array([0, 0, 200], np.uint8)
            color_high = np.array([180, 25, 255], np.uint8)
        case Color.ORANGE:
            color_low = np.array([10, 80, 180], np.uint8)
            color_high = np.array([14, 170, 200], np.uint8)
        case Color.RED:
            color_low = np.array([10, 70, 20], np.uint8)
            color_high = np.array([20, 100, 160], np.uint8)
        case Color.BROWN:
            color_low = np.array([0, 0, 0], np.uint8)
            color_high = np.array([255, 255, 255], np.uint8)
        case Color.YELLOW:
            color_low = np.array([0, 0, 0], np.uint8)
            color_high = np.array([255, 255, 255], np.uint8)
        case Color.GREEN:
            color_low = np.array([70, 30, 70], np.uint8)
            color_high = np.array([80, 50, 110], np.uint8)
        case Color.BLUE:
            color_low = np.array([100, 100, 100], np.uint8)
            color_high = np.array([130, 200, 200], np.uint8)
        case Color.BLACK:
            color_low = np.array([0, 45, 90], np.uint8)
            color_high = np.array([20, 70, 110], np.uint8)
    return cv2.dilate(cv2.inRange(image_hsv, color_low, color_high), clean_image)


def get_types():
    return [Type.Ball,
            Type.OrangeBall,
            Type.Wall,
            Type.Wall,
            Type.Robot,
            Type.BigGreenGoal,
            Type.SmallBlueGoal]


def get_masks(image_hsv):
    clean_image = np.ones((5, 5), "uint8")
    return [mask_from_color(Color.WHITE, image_hsv, clean_image),
            mask_from_color(Color.ORANGE, image_hsv, clean_image),
            mask_from_color(Color.RED, image_hsv, clean_image),
            mask_from_color(Color.BROWN, image_hsv, clean_image),
            mask_from_color(Color.BLACK, image_hsv, clean_image),
            mask_from_color(Color.GREEN, image_hsv, clean_image),
            mask_from_color(Color.BLUE, image_hsv, clean_image)]


def reset():
    ballsState.clear()
    global robot_state
    robot_state = Robot([0, 0], [0, 0], [0, 0])


def draw_robot(image):
    cv2.rectangle(image, (600, 250), (600 + 50, 250 + 50),
                  (Type.Robot.value[0], Type.Robot.value[1], Type.Robot.value[2]), 2)




# Hex to BGR converter
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))


def vector_between_points(point1, point2):
    """ Calculate the vector between two points (x, y). """
    return np.array([point1[0] - point2[0], point1[1] - point2[1]])


# Initialize EMA variables
prev_middle_x = None
prev_middle_y = None
alpha = 0.2  # Smoothing factor

# History buffers for running average
history_length = 10
low_x_history = deque(maxlen=50)
high_x_history = deque(maxlen=50)
middle_points_history = deque(maxlen=100)

# Grids for path
grid_row = 75
grid_col = 75

glob_grid = None

real_world_distance = shared_state.real_world_distance


def all_grid_calc():
    ret, image = cam.read()
    global glob_grid
    cell_height, cell_width, glob_grid = initialize_grid(image, grid_row, grid_col)
    update_grid_with_obstacles(image, glob_grid, cell_height, cell_width)
    shared_state.current_cell_height = cell_height
    shared_state.current_cell_width = cell_width
    shared_state.current_grid = glob_grid


'''New Grid and obstacle addition Starts here'''
'''Create a grid over the image'''
def initialize_grid(image, grid_rows, grid_cols):
    h, w = image.shape[:2]
    cell_height = h / grid_rows
    cell_width = w / grid_cols
    return cell_height, cell_width, [[0] * grid_cols for _ in range(grid_rows)]

'''Check each grid for obstacles (wall and egg) and mark the grid_slot as 1 if there is an obstacle'''
'''This is a bit costly, but since obstacle shouldnt move, can it be be run less frequently'''
def update_grid_with_obstacles(image, grid, cell_height, cell_width):
    wall_bgr_color = hex_to_bgr("ff5c0d")
    egg_bgr_color = hex_to_bgr("FDF7F5")

    wall_lower_bound = np.array([max(c - 80, 0) for c in wall_bgr_color])
    wall_upper_bound = np.array([min(c + 80, 255) for c in wall_bgr_color])

    egg_lower_bound = np.array([max(c - 20, 0) for c in egg_bgr_color])
    egg_upper_bound = np.array([min(c + 20, 255) for c in egg_bgr_color])

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            # Define how many grids around we look inside as well
            top_left = (int(max((j - 1) * cell_width, 0)), int(max((i - 1) * cell_height, 0)))
            bottom_right = (int(min((j + 2) * cell_width, image.shape[1])), int(min((i + 2) * cell_height, image.shape[0])))
            cell_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

            # Create a mask for the red wall color range
            wall_mask = cv2.inRange(cell_image, wall_lower_bound, wall_upper_bound)

            egg_mask = cv2.inRange(cell_image, egg_lower_bound, egg_upper_bound)

            # Find contours in the cell image
            wall_contours, _ = cv2.findContours(wall_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            egg_contours, _ = cv2.findContours(egg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # If any contours are found and big enough, mark the grid cell
            for(contour) in  wall_contours:
                if 100 <= cv2.contourArea(contour): #Adjust these numbers as needed
                    grid[i][j] = 1

            for(contour) in egg_contours:
                if 100 <= cv2.contourArea(contour): #Adjust these numbers as needed
                    grid[i][j] = 1


'''Helper function, it is used to visualize the grid on the image'''
def draw_grid(image, grid, cell_height, cell_width):
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            top_left = (int(col * cell_width), int(row * cell_height))
            bottom_right = (int((col + 1) * cell_width), int((row + 1) * cell_height))
            if grid[row][col] == 1:
                #print("Wall at grid col: ", col, " and row: ", row)
                cv2.rectangle(image, top_left, bottom_right, (255,69,0), -1)  # Fill with blue if there's a wall
            cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), 1)  # Draw the grid line

'''Newly added wall and grid ends here'''



def vector_intersects_box_with_image(image, robot_position, vector, box_center, box_width, robot_width):
    box_left = int(box_center[0] - box_width // 2)
    box_right = int(box_center[0] + box_width // 2)
    box_top = int(box_center[1] - box_width // 2)
    box_bottom = int(box_center[1] + box_width // 2)

    def line_intersects_line(p1, p2, p3, p4):
        # Check if line segments (p1, p2) and (p3, p4) intersect
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    # Calculate the target position based on the vector
    target_position = (int(robot_position[0] - vector[0]), int(robot_position[1] - vector[1]))

    # Calculate perpendicular offset for the robot's width
    vector_magnitude = np.sqrt(vector[0]**2 + vector[1]**2)
    offset_x = (robot_width / 2) * (vector[1] / vector_magnitude)
    offset_y = (robot_width / 2) * (vector[0] / vector_magnitude)

    # Calculate the positions of the parallel lines (edges of the robot)
    left_robot_position = (robot_position[0] - offset_x, robot_position[1] + offset_y)
    right_robot_position = (robot_position[0] + offset_x, robot_position[1] - offset_y)
    left_target_position = (target_position[0] - offset_x, target_position[1] + offset_y)
    right_target_position = (target_position[0] + offset_x, target_position[1] - offset_y)

    # Define the corners of the bounding box
    top_left = (box_left, box_top)
    top_right = (box_right, box_top)
    bottom_left = (box_left, box_bottom)
    bottom_right = (box_right, box_bottom)

    # Debug print statements
    print(f"Robot Position: {robot_position}")
    print(f"Target Position: {target_position}")
    print(f"Bounding Box: {top_left}, {top_right}, {bottom_left}, {bottom_right}")

    # Draw the bounding box
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)  # Green box

    # Draw the robot's paths
    cv2.line(image, tuple(map(int, left_robot_position)), tuple(map(int, left_target_position)), (255, 0, 0), 2)  # Blue line for the left edge
    cv2.line(image, tuple(map(int, right_robot_position)), tuple(map(int, right_target_position)), (255, 0, 0), 2)  # Blue line for the right edge

    # Check if either of the lines intersects any of the box's sides
    if (line_intersects_line(left_robot_position, left_target_position, top_left, top_right) or
            line_intersects_line(left_robot_position, left_target_position, top_right, bottom_right) or
            line_intersects_line(left_robot_position, left_target_position, bottom_right, bottom_left) or
            line_intersects_line(left_robot_position, left_target_position, bottom_left, top_left) or
            line_intersects_line(right_robot_position, right_target_position, top_left, top_right) or
            line_intersects_line(right_robot_position, right_target_position, top_right, bottom_right) or
            line_intersects_line(right_robot_position, right_target_position, bottom_right, bottom_left) or
            line_intersects_line(right_robot_position, right_target_position, bottom_left, top_left)):
        print("The robot's path intersects with the cross.")
        return True

    print("The robot's path does not intersect with the cross.")
    return False


def find_next_safe_point_with_image(image, robot_position, ball_position, box_center, box_width, robot_width):
    # Access cross positions directly from shared_state
    cross_positions = {
        'top_left': shared_state.cross_top_left,
        'top_right': shared_state.cross_top_right,
        'bottom_left': shared_state.cross_bottom_left,
        'bottom_right': shared_state.cross_bottom_right
    }

    # Calculate the distances from the ball to each cross position
    distances_to_ball = {
        'top_left': np.linalg.norm(np.array(ball_position) - np.array(cross_positions['top_left'])),
        'top_right': np.linalg.norm(np.array(ball_position) - np.array(cross_positions['top_right'])),
        'bottom_left': np.linalg.norm(np.array(ball_position) - np.array(cross_positions['bottom_left'])),
        'bottom_right': np.linalg.norm(np.array(ball_position) - np.array(cross_positions['bottom_right'])),
    }

    # Sort the cross positions by distance to the ball
    sorted_cross_positions = sorted(distances_to_ball.items(), key=lambda item: item[1])

    # Find the next safe point for the robot
    for pos_name, _ in sorted_cross_positions:
        safe_point = cross_positions[pos_name]

        # Calculate the vector for the robot's path
        vector_to_safe_point = vector_between_points(robot_position, safe_point)



        # Check if the path to the safe point intersects with the cross box
        if not vector_intersects_box_with_image(image, robot_position, vector_to_safe_point, box_center, box_width, robot_width):
            print(cross_positions)
            print("Next point to go to: ", safe_point)
            return safe_point

    return None  # Return None if no safe point is found



# Function to convert hex color to HSV
def hex_to_hsv(hex_color):
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    rgb_color = tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))
    bgr_color = np.uint8([[list(rgb_color[::-1])]])
    hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0][0]
    return hsv_color




def detect_multiple_colors_in_image(image, colors):
    ball_positions = []
    robot_positions = []
    goal_position = None
    wall_positions = []
    wall_x_positions = []
    highest_x_point = None
    lowest_x_point = None
    cross_positions = []
    wall_y_positions = []
    low_y_history = []
    high_y_history = []




    # Assuming 'image' is the image array
    height, width, _ = image.shape
    middle_x = width // 2
    middle_y = height // 2




    for color in colors:
        bgr_color = hex_to_bgr(color['hex_color'])
        lower_bound = np.array([max(c - color['tolerance'], 0) for c in bgr_color])
        upper_bound = np.array([min(c + color['tolerance'], 255) for c in bgr_color])

        # Create a mask for the color range
        mask = cv2.inRange(image, lower_bound, upper_bound)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Draw contours around detected areas and record positions
        for contour in contours:
            area = cv2.contourArea(contour)
            min_area = color.get('min_area', 0)
            max_area = color.get('max_area', float('inf'))
            if min_area < area < max_area:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cX = int(M['m10'] / M['m00'])
                    cY = int(M['m01'] / M['m00'])
                    if color['name'] == 'balls':
                        ball_positions.append((cX, cY))
                    elif color['name'] == 'robot':
                        robot_positions.append((cX, cY))
                    elif color['name'] == 'goal':
                        goal_position = (cX, cY)
                    elif color['name'] == 'wall':
                        wall_positions.append((cX, cY))
                        # Append all x-values from the contour to wall_x_positions
                        for point in contour:
                            x, y = point[0]
                            if middle_x - 80 <= x <= middle_x + 80 and middle_y - 80 <= y <= middle_y + 80:
                                cross_positions.append((x, y))
                            else:
                                wall_x_positions.append(x)
                                wall_y_positions.append(y)
                            #print("Point: ", point)
                cv2.drawContours(image, [contour], -1, color['draw_color'], 2)

    # Draw circles for detected ball and robot positions
    for pos in ball_positions:
        cv2.circle(image, pos, 5, (0, 0, 0), -1)  # Black circle for balls

    for pos in robot_positions:
        cv2.circle(image, pos, 5, (0, 0, 255), -1)  # Red circle for robots

    if not ball_positions:
        print("No balls detected.")
    if not robot_positions:
        print("No robots detected.")
    if goal_position is None:
        print("No goal detected.")

    # Calculate the middle point of all wall contours
    if wall_positions:
        wall_positions = np.array(wall_positions)
        avg_x = int(np.mean(wall_positions[:, 0]))
        avg_y = int(np.mean(wall_positions[:, 1]))
        middle_point = (avg_x, avg_y)

        # Append the new middle point to the history
        middle_points_history.append(middle_point)

        # Calculate the most common middle point from history for stability
        most_common_middle_point = Counter(middle_points_history).most_common(1)[0][0]

        shared_state.middlepoint = most_common_middle_point

        #print("Most common middlepoint: ", most_common_middle_point)

        cv2.circle(image, most_common_middle_point, 5, (0, 0, 0), -1)

    lowest_x_with_center_y = None
    highest_x_with_center_y = None

    # Filter x-values for the desired ranges
    low_x_values = [x for x in wall_x_positions if x < 100]
    high_x_values = [x for x in wall_x_positions if x > 500]

    # Find the most common x-value in each range
    if low_x_values:
        most_common_low_x = Counter(low_x_values).most_common(1)[0][0]
        low_x_history.append(most_common_low_x)
        stable_low_x = int(np.mean(low_x_history))
        lowest_x_with_center_y = (stable_low_x, most_common_middle_point[1])
        shared_state.left_wall = stable_low_x
        shared_state.low_x = lowest_x_with_center_y
        cv2.circle(image, lowest_x_with_center_y, 5, (0, 0, 0), -1)  # Black dot for the most common low x with center y

    if high_x_values:
        most_common_high_x = Counter(high_x_values).most_common(1)[0][0]
        high_x_history.append(most_common_high_x)
        stable_high_x = int(np.mean(high_x_history))
        highest_x_with_center_y = (stable_high_x, most_common_middle_point[1])
        shared_state.right_wall = stable_high_x
        cv2.circle(image, highest_x_with_center_y, 5, (0, 0, 0), -1)  # Black dot for the most common high x with center y


    # Filter y-values for the desired ranges
    low_y_values = [y for y in wall_y_positions if y < 100]
    high_y_values = [y for y in wall_y_positions if y > 300]

    # Find the most common y-value in each range
    if low_y_values:
        most_common_low_y = Counter(low_y_values).most_common(1)[0][0]
        low_y_history.append(most_common_low_y)
        stable_low_y = int(np.mean(low_y_history))
        lowest_y_with_center_x = (most_common_middle_point[0], stable_low_y)
        shared_state.upper_wall = stable_low_y
        shared_state.low_y = lowest_y_with_center_x
        cv2.circle(image, lowest_y_with_center_x, 5, (0, 0, 0), -1)  # Black dot for the most common low y with center x

    if high_y_values:
        most_common_high_y = Counter(high_y_values).most_common(1)[0][0]
        high_y_history.append(most_common_high_y)
        stable_high_y = int(np.mean(high_y_history))
        highest_y_with_center_x = (most_common_middle_point[0], stable_high_y)
        shared_state.lower_wall = stable_high_y
        shared_state.high_y = highest_y_with_center_x
        cv2.circle(image, highest_y_with_center_x, 5, (0, 0, 0), -1)

        # Calculate the reference vector magnitude
    if lowest_x_with_center_y and highest_x_with_center_y:
        reference_vector = vector_between_points(highest_x_with_center_y, lowest_x_with_center_y)
        shared_state.reference_vector_magnitude = np.linalg.norm(reference_vector)

    if cross_positions:
        cross_positions = np.array(cross_positions)
        avg_cross_x = int(np.mean(cross_positions[:, 0]))
        avg_cross_y = int(np.mean(cross_positions[:, 1]))
        middle_cross_point = (avg_cross_x, avg_cross_y)

        shared_state.cross_middle = middle_cross_point

        # Define offsets for additional points
        offsets = [(120, 120), (120, -120), (-120, 120), (-120, -120)]

        # Initialize variables to store the cross corners
        cross_top_left = None
        cross_top_right = None
        cross_bottom_left = None
        cross_bottom_right = None

        # Draw additional points with the offsets and update shared_state
        for i, (dx, dy) in enumerate(offsets):
            offset_point = (middle_cross_point[0] + dx, middle_cross_point[1] + dy)
            cv2.circle(image, offset_point, 5, (0, 255, 0), -1)  # Green circles for offset points

            # Assign the offset points to the respective cross corner
            if dx == -120 and dy == -120:
                cross_top_left = offset_point
            elif dx == 120 and dy == -120:
                cross_top_right = offset_point
            elif dx == -120 and dy == 120:
                cross_bottom_left = offset_point
            elif dx == 120 and dy == 120:
                cross_bottom_right = offset_point

                # Update shared_state with the cross corners
        shared_state.cross_top_left = cross_top_left
        shared_state.cross_top_right = cross_top_right
        shared_state.cross_bottom_left = cross_bottom_left
        shared_state.cross_bottom_right = cross_bottom_right
        cv2.circle(image, middle_cross_point, 5, (0, 255, 0), -1)



    ball_positions = ball_positions[:10]
    robot_positions = robot_positions[:3]

    # Draw circles at the four spots around the image center
    #offsets = [(80, 80), (80, -80), (-80, 80), (-80, -80)]
    #for dx, dy in offsets:
    #cv2.circle(image, (middle_x + dx, middle_y + dy), 5, (255, 0, 255), -1)  # Purple circles


    # Draw a circle at the exact middle of the image
    cv2.circle(image, (middle_x, middle_y), 5, (0, 255, 0), -1)  # Green circle for the middle point

    # Draw circles for cross positions
    #for pos in cross_positions:
    #   cv2.circle(image, pos, 5, (255, 0, 0), -1)  # Purple circle for cross positions




    #temp_vector = vector_between_points(shared_state.real_position_robo, ball_positions[0])

    #vector_intersects_box_with_image(image, shared_state.real_position_robo, temp_vector, shared_state.cross_middle, 60, 45)


    #find_next_safe_point_with_image(image, shared_state.real_position_robo, ball_positions[0], shared_state.cross_middle, 60, 40)



    return ball_positions, robot_positions, goal_position

# Define colors and their properties
colors = [
    {
        'name': 'balls',
        'hex_color': 'FDF9F8',
        'tolerance': 60,
        'min_area': 40,
        'max_area': 200,
        'draw_color': (255, 255, 255)  # Green
    },
    {
        'name': 'egg',
        'hex_color': 'FDF7F5',
        'tolerance': 20,
        'min_area': 300,
        'max_area': 1000,
        'draw_color': (0, 0, 255)
    },
    {
        'name': 'wall',
        'hex_color': 'F03A26',
        'tolerance': 70,
        'min_area': 200,
        'max_area': 15000000,
        'draw_color': (255, 0, 255)  # Purple
    },
    {
        'name': 'robot',
        'hex_color': '35CCC6',
        'tolerance': 45,
        'min_area': 400,
        'max_area': 1500,
        'draw_color': (255, 0, 0)  # Blue
    },
    {
        'name': 'goal',
        'hex_color': 'FEFFAB',
        'tolerance': 12,
        'min_area': 30,
        'max_area': 300,
        'draw_color': (0, 0, 0)  # Black
    },
    {
        'name': 'orange_balls',
        'hex_color': 'FE9546',
        'tolerance': 30,
        'min_area': 50,
        'max_area': 300,
        'draw_color': (0, 255, 255)
    }
]


def calculate_distance(point1, point2):
    """ Calculate the Euclidean distance between two points. """
    return np.linalg.norm(np.array(point1) - np.array(point2))



# Given values
robot_real_height = 17.0  # cm
camera_height = 187  # cm
field = 84


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

    image_height = shared_state.image_height

    # The x-coordinate is unaffected in this simplified top-down projection
    real_world_x = robot_pos[0]

    return real_world_x, real_world_y

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


def robot_rotation(position, image_height):
    # Convert the front and back positions to real-world coordinates
    front_real_world = calculate_real_world_position(position[0], image_height)
    back_real_world = calculate_real_world_position(position[1], image_height)
    return angle_of_vector(front_real_world[0] - back_real_world[0], front_real_world[1] - back_real_world[1])


def robot_rotation_old(position):
    return angle_of_vector(position[0][0]-position[1][0], position[0][1] - position[1][1])

def calculate_pixel_to_cm_ratio(pixel_distance):
    half_field_width_cm = 84  # Known half-field width in cm
    return half_field_width_cm / pixel_distance

def calculate_b_adj(camera_height, robot_height, horizontal_distance_cm):
    effective_height = camera_height - robot_height
    return horizontal_distance_cm * (effective_height / camera_height)

def reverse_angle(angle):
    # Add 180 to the given angle and take modulo 360 to ensure it's within 0-360
    reversed_angle = (angle + 180) % 360
    return reversed_angle

def calculate_final_position(midpoint, robot_position, camera_height, end_of_field, robot_height):
    pixel_distance = math.sqrt((end_of_field[0] - midpoint[0])**2 + (end_of_field[1] - midpoint[1])**2)
    shared_state.half_field_pixel = pixel_distance
    pixel_to_cm_ratio = calculate_pixel_to_cm_ratio(pixel_distance)
    dx = robot_position[0] - midpoint[0]
    dy = robot_position[1] - midpoint[1]
    horizontal_distance_cm = math.sqrt(dx**2 + dy**2) * pixel_to_cm_ratio



    b_adj = calculate_b_adj(camera_height, robot_height, horizontal_distance_cm)

    #print("B in cm: ", horizontal_distance_cm)
    #print("b_adj: ", b_adj)

    angle = angle_of_vector(dx, dy)
    reversed_angle = reverse_angle(angle)


    #print("Angle of vector: ", reversed_angle)
    rad_angle = np.deg2rad(reversed_angle)
    #print("Rad angle: ", rad_angle)
    #print("back to pixels: ", b_adj/pixel_to_cm_ratio)
    #print("84 cm in pixels: ", pixel_distance)

    robo_pix_distance = b_adj/pixel_to_cm_ratio

    final_x = midpoint[0] - robo_pix_distance * math.cos(rad_angle)
    final_y = midpoint[1] + robo_pix_distance * math.sin(rad_angle)

    return (final_x, final_y)


print(calculate_b_adj(187.5, 16, 84))

def render():
    reset()  # Assuming reset() is defined elsewhere
    ret, image = cam.read()  # Reading Images
    #image = adjust_exposure(image)  # Adjust these parameters as needed
    #image = enhance_contrast(image)
    shared_state.image = image
    if not ret:
        print("Error: Failed to read frame.")
        return None

    # Detect multiple colors in the image
    ball_positions, robot_positions, goal_position = detect_multiple_colors_in_image(image, colors)

    if len(robot_positions) != 3:
        print("Are we here")
        cv2.imshow('Frame', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
        return None

    state = State(
        balls=[Ball(x, y, True) for x, y in ball_positions],
        corners=[],  # Update this if you need corners
        robot=Robot(*robot_positions[:3]),
        small_goal_pos=None,  # Update this if you have small_goal_pos
        big_goal_pos=goal_position  # Update this if you have big_goal_pos
    )

    # Calculate front and back positions of the robot
    front_and_back = robot_front_and_back(state.robot)

    #print("Robot rotation: ", robot_rotation(front_and_back, shared_state.image_height))

    rob_rot = robot_rotation_old(front_and_back)

    #print("Old rot: ", rob_rot)

    if front_and_back is not None:
        # Calculate robot position and draw a blue circle
        robot_pos = robot_position(front_and_back)

        cv2.circle(image, (int(robot_pos[0]), int(robot_pos[1])), 5, (255, 0, 0), -1)

        coords = calculate_final_position(shared_state.middlepoint, robot_pos, camera_height, shared_state.low_x, robot_real_height)
        #print("Start robo: ", robot_pos)
        #print("Middle point: ", shared_state.middlepoint[0], " ", shared_state.middlepoint[1])
        #print("My coords: ", coords)
        cv2.circle(image, (int(coords[0]), int(coords[1])), 5, (255, 255, 255), -1)



        shared_state.real_position_robo = coords

    if goal_position is not None:
        robot_center = robot_positions[0]  # Use the first robot position as the center
        distance_to_goal = calculate_distance(robot_center, goal_position)

    calculate_final_position



    # Display the frame with contours and circles
    cv2.imshow('Frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

    return state



def adjust_exposure(image):
    alpha = 1    #param alpha: Contrast control. 1.0-3.0 for more contrast, <1.0 for less.
    beta = -50     #param beta: Brightness control. 0-100 for brighter, <0 for darker.

    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def enhance_contrast(image):
    """
    Enhances the contrast of the image using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    :param image: Input image.
    :return: Contrast-enhanced image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
