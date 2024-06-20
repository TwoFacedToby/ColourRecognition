import numpy as np
import cv2
from enum import Enum
from scipy.stats import mode
from collections import Counter
from scipy.stats import circmean
from MovementController import next_command_from_state, robot_position, robot_front_and_back, shortest_vector_with_index, vector_from_robot_to_next_ball
from sklearn.cluster import DBSCAN
from collections import Counter, deque
import shared_state
import math

# Capturing video through webcam
cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus

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

def initialize_grid(image, grid_rows, grid_cols):
    h, w = image.shape[:2]
    cell_height = h / grid_rows
    cell_width = w / grid_cols
    return cell_height, cell_width, [[0] * grid_cols for _ in range(grid_rows)]

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

def draw_grid(image, grid, cell_height, cell_width):
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            top_left = (int(col * cell_width), int(row * cell_height))
            bottom_right = (int((col + 1) * cell_width), int((row + 1) * cell_height))
            if grid[row][col] == 1:
                #print("Wall at grid col: ", col, " and row: ", row)
                cv2.rectangle(image, top_left, bottom_right, (255,69,0), -1)  # Fill with blue if there's a wall
            cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), 1)  # Draw the grid line

def vector_between_points(point1, point2):
    """ Calculate the vector between two points (x, y). """
    return np.array([point1[0] - point2[0], point1[1] - point2[1]])

def reset():
    ballsState.clear()
    global robot_state
    robot_state = Robot([0, 0], [0, 0], [0, 0])

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


def mask_from_color(h_lower, h_upper, s_lower, s_upper, v_lower, v_upper, image_hsv, clean_image):
    lower_bound = np.array([h_lower, s_lower, v_lower], np.uint8)
    upper_bound = np.array([h_upper, s_upper, v_upper], np.uint8)
    return cv2.dilate(cv2.inRange(image_hsv, lower_bound, upper_bound), clean_image)

def get_colors():
    return [
        {
            'name': 'balls',
            'hex_color': 'FEFDFD',
            'h_lower': 0,
            'h_upper': 164,
            's_lower': 0,
            's_upper': 18,
            'v_lower': 201,
            'v_upper': 255,
            'min_area': 40,
            'max_area': 200,
            'draw_color': (255, 255, 255)
        },
        {
            'name': 'egg',
            'hex_color': 'FDF7F5',
            'h_lower': 0,
            'h_upper': 164,
            's_lower': 0,
            's_upper': 63,
            'v_lower': 201,
            'v_upper': 255,
            'min_area': 300,
            'max_area': 1000,
            'draw_color': (0, 0, 255)
        },
        {
            'name': 'wall',
            'hex_color': 'F03A26',
            'h_lower': 0,
            'h_upper': 8,
            's_lower': 80,
            's_upper': 255,
            'v_lower': 0,
            'v_upper': 255,
            'min_area': 200,
            'max_area': 0,
            'draw_color': (255, 0, 255)
        },
        {
            'name': 'robot',
            'hex_color': '35CCC6',
            'h_lower': 22,
            'h_upper': 128,
            's_lower': 90,
            's_upper': 255,
            'v_lower': 81,
            'v_upper': 255,
            'min_area': 400,
            'max_area': 1500,
            'draw_color': (255, 0, 0)
        },
        {
            'name': 'goal',
            'hex_color': 'FEFFAB',
            'h_lower': 21,
            'h_upper': 51,
            's_lower': 52,
            's_upper': 94,
            'v_lower': 207,
            'v_upper': 255,
            'min_area': 30,
            'max_area': 600,
            'draw_color': (0, 0, 0)
        },
        {
            'name': 'orange_balls',
            'hex_color': 'FE9546',
            'h_lower': 13,
            'h_upper': 44,
            's_lower': 87,
            's_upper': 212,
            'v_lower': 220,
            'v_upper': 255,
            'min_area': 70,
            'max_area': 300,
            'draw_color': (0, 255, 255)
        }
    ]

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
    middle_points_history = deque(maxlen=100)
    low_x_history = deque(maxlen=50)
    high_x_history = deque(maxlen=50)
    middle_cross_point = None  # Initialize middle_cross_point

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width, _ = hsv_image.shape
    middle_x = width // 2
    middle_y = height // 2

    for color in colors:
        lower_bound = np.array([color['h_lower'], color['s_lower'], color['v_lower']])
        upper_bound = np.array([color['h_upper'], color['s_upper'], color['v_upper']])

        # Create a mask for the color range
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Draw contours around detected areas and record positions
        for contour in contours:
            area = cv2.contourArea(contour)
            min_area = color.get('min_area', 0)
            max_area = color.get('max_area', float('inf'))
            if max_area == 0:  # Treat max_area 0 as infinite
                max_area = float('inf')
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
                        for point in contour:
                            x, y = point[0]
                            if middle_x - 80 <= x <= middle_x + 80 and middle_y - 80 <= y <= middle_y + 80:
                                cross_positions.append((x, y))
                            else:
                                wall_x_positions.append(x)
                                wall_y_positions.append(y)
                cv2.drawContours(image, [contour], -1, color['draw_color'], 2)

    for pos in ball_positions:
        cv2.circle(image, pos, 5, (0, 0, 0), -1)

    for pos in robot_positions:
        cv2.circle(image, pos, 5, (0, 0, 255), -1)

    if wall_positions:
        wall_positions = np.array(wall_positions)
        avg_x = int(np.mean(wall_positions[:, 0]))
        avg_y = int(np.mean(wall_positions[:, 1]))
        middle_point = (avg_x, avg_y)
        middle_points_history.append(middle_point)
        most_common_middle_point = Counter(middle_points_history).most_common(1)[0][0]
        shared_state.middlepoint = most_common_middle_point
        cv2.circle(image, most_common_middle_point, 5, (0, 0, 0), -1)

    lowest_x_with_center_y = None
    highest_x_with_center_y = None

    low_x_values = [x for x in wall_x_positions if x < 100]
    high_x_values = [x for x in wall_x_positions if x > 500]

    if low_x_values:
        most_common_low_x = Counter(low_x_values).most_common(1)[0][0]
        low_x_history.append(most_common_low_x)
        stable_low_x = int(np.mean(low_x_history))
        lowest_x_with_center_y = (stable_low_x, most_common_middle_point[1])
        shared_state.left_wall = stable_low_x
        shared_state.low_x = lowest_x_with_center_y
        cv2.circle(image, lowest_x_with_center_y, 5, (0, 0, 0), -1)

    if high_x_values:
        most_common_high_x = Counter(high_x_values).most_common(1)[0][0]
        high_x_history.append(most_common_high_x)
        stable_high_x = int(np.mean(high_x_history))
        highest_x_with_center_y = (stable_high_x, most_common_middle_point[1])
        shared_state.right_wall = stable_high_x
        cv2.circle(image, highest_x_with_center_y, 5, (0, 0, 0), -1)

    low_y_values = [y for y in wall_y_positions if y < 100]
    high_y_values = [y for y in wall_y_positions if y > 300]

    if low_y_values:
        most_common_low_y = Counter(low_y_values).most_common(1)[0][0]
        low_y_history.append(most_common_low_y)
        stable_low_y = int(np.mean(low_y_history))
        lowest_y_with_center_x = (most_common_middle_point[0], stable_low_y)
        shared_state.upper_wall = stable_low_y
        shared_state.low_y = lowest_y_with_center_x
        cv2.circle(image, lowest_y_with_center_x, 5, (0, 0, 0), -1)

    if high_y_values:
        most_common_high_y = Counter(high_y_values).most_common(1)[0][0]
        high_y_history.append(most_common_high_y)
        stable_high_y = int(np.mean(high_y_history))
        highest_y_with_center_x = (most_common_middle_point[0], stable_high_y)
        shared_state.lower_wall = stable_high_y
        shared_state.high_y = highest_y_with_center_x
        cv2.circle(image, highest_y_with_center_x, 5, (0, 0, 0), -1)

    if lowest_x_with_center_y and highest_x_with_center_y:
        reference_vector = vector_between_points(highest_x_with_center_y, lowest_x_with_center_y)
        shared_state.reference_vector_magnitude = np.linalg.norm(reference_vector)

    if cross_positions:
        cross_positions = np.array(cross_positions)
        avg_cross_x = int(np.mean(cross_positions[:, 0]))
        avg_cross_y = int(np.mean(cross_positions[:, 1]))
        middle_cross_point = (avg_cross_x, avg_cross_y)
        shared_state.cross_middle = middle_cross_point
        offsets = [(120, 120), (120, -120), (-120, 120), (-120, -120)]
        cross_top_left = None
        cross_top_right = None
        cross_bottom_left = None
        cross_bottom_right = None

        for i, (dx, dy) in enumerate(offsets):
            offset_point = (middle_cross_point[0] + dx, middle_cross_point[1] + dy)
            cv2.circle(image, offset_point, 5, (0, 255, 0), -1)
            if dx == -120 and dy == -120:
                cross_top_left = offset_point
            elif dx == 120 and dy == -120:
                cross_top_right = offset_point
            elif dx == -120 and dy == 120:
                cross_bottom_left = offset_point
            elif dx == 120 and dy == 120:
                cross_bottom_right = offset_point

        shared_state.cross_top_left = cross_top_left
        shared_state.cross_top_right = cross_top_right
        shared_state.cross_bottom_left = cross_bottom_left
        shared_state.cross_bottom_right = cross_bottom_right
        cv2.circle(image, middle_cross_point, 5, (0, 255, 0), -1)

    ball_positions = ball_positions[:10]
    robot_positions = robot_positions[:3]

    cv2.circle(image, (middle_x, middle_y), 5, (0, 255, 0), -1)

    return ball_positions, robot_positions, goal_position




# Define colors and their properties
colors = get_colors()

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
    if midpoint is None or end_of_field is None:
        return robot_position  # or handle as needed

    pixel_distance = math.sqrt((end_of_field[0] - midpoint[0])**2 + (end_of_field[1] - midpoint[1])**2)
    shared_state.half_field_pixel = pixel_distance
    pixel_to_cm_ratio = calculate_pixel_to_cm_ratio(pixel_distance)
    dx = robot_position[0] - midpoint[0]
    dy = robot_position[1] - midpoint[1]
    horizontal_distance_cm = math.sqrt(dx**2 + dy**2) * pixel_to_cm_ratio

    b_adj = calculate_b_adj(camera_height, robot_height, horizontal_distance_cm)

    angle = angle_of_vector(dx, dy)
    reversed_angle = reverse_angle(angle)

    rad_angle = np.deg2rad(reversed_angle)

    robo_pix_distance = b_adj/pixel_to_cm_ratio

    final_x = midpoint[0] - robo_pix_distance * math.cos(rad_angle)
    final_y = midpoint[1] + robo_pix_distance * math.sin(rad_angle)

    return (final_x, final_y)


def render():
    reset()
    ret, image = cam.read()
    shared_state.image = image
    if not ret:
        print("Error: Failed to read frame.")
        return None

    ball_positions, robot_positions, goal_position = detect_multiple_colors_in_image(image, colors)

    if len(robot_positions) != 3:
        cv2.imshow('Frame', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
        return None

    state = State(
        balls=[Ball(x, y, True) for x, y in ball_positions],
        corners=[],
        robot=Robot(*robot_positions[:3]),
        small_goal_pos=None,
        big_goal_pos=goal_position
    )

    front_and_back = robot_front_and_back(state.robot)
    rob_rot = robot_rotation_old(front_and_back)

    if front_and_back is not None:
        robot_pos = robot_position(front_and_back)
        cv2.circle(image, (int(robot_pos[0]), int(robot_pos[1])), 5, (255, 0, 0), -1)

        if shared_state.middlepoint:
            coords = calculate_final_position(shared_state.middlepoint, robot_pos, camera_height, shared_state.low_x, robot_real_height)
            cv2.circle(image, (int(coords[0]), int(coords[1])), 5, (255, 255, 255), -1)
            shared_state.real_position_robo = coords

    if goal_position is not None:
        robot_center = robot_positions[0]
        distance_to_goal = calculate_distance(robot_center, goal_position)

    cv2.imshow('Frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

    return state



def adjust_exposure(image):
    alpha = 1    #param alpha: Contrast control. 1.0-3.0 for more contrast, <1.0 for less.
    beta = -50     #param beta: Brightness control. 0-100 for brighter, <0 for darker.

    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Hex to BGR converter
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))


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
