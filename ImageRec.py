import numpy as np
import cv2
from enum import Enum
import matplotlib.pyplot as plt
from scipy.stats import mode
from collections import Counter
from scipy.stats import circmean
from MovementController import next_command_from_state
import heapq
# Capturing video through webcam
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)


#cam = cv2.VideoCapture("TrackVideos/Test_vid.mkv")

class State:
    def __init__(self, balls, corners, robot, small_goal_pos, big_goal_pos, grid):
        self.balls = balls
        self.corners = corners
        self.robot = robot
        self.small_goal_pos = small_goal_pos
        self.big_goal_pos = big_goal_pos
        self.grid = grid


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

class Grid_tile:
    def __init__(self, pos_1, pos_2, has_wall):
        self.pos_1 = pos_1
        self.pos_2 = pos_2
        self.has_wall = has_wall


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

def initialize_grid(image, grid_rows, grid_cols):
    h, w = image.shape[:2]
    cell_height = h / grid_rows
    cell_width = w / grid_cols
    return cell_height, cell_width, [[0] * grid_cols for _ in range(grid_rows)]

'''Doesnt work
def update_grid_with_walls(grid, contours, cell_height, cell_width):
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            # Define the bounding box for the current grid cell
            top_left = (j * cell_width, i * cell_height)
            bottom_right = ((j + 1) * cell_width, (i + 1) * cell_height)
            grid_cell_box = np.array([top_left, (top_left[0], bottom_right[1]), bottom_right, (bottom_right[0], top_left[1])])

            # Check if any contour points are inside the grid cell bounding box
            for contour in contours:
                if cv2.pointPolygonTest(grid_cell_box, tuple(contour[0][0]), False) >= 0:
                    grid[i][j] = 1
                    break  # No need to check further contours for this cell
                    '''
def draw_grid(image, grid, cell_height, cell_width):
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            top_left = (int(col * cell_width), int(row * cell_height))
            bottom_right = (int((col + 1) * cell_width), int((row + 1) * cell_height))
            if grid[row][col] == 1:
                print("Wall at grid col: ", col, " and row: ", row)
                cv2.rectangle(image, top_left, bottom_right, (255,69,0), -1)  # Fill with blue if there's a wall
            cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), 1)  # Draw the grid line

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

def detect_multiple_colors_in_image(image, colors):
    ball_positions = []
    robot_positions = []
    goal_position = None
    walls = []

    cell_height, cell_width, grid = initialize_grid(image, 75, 75)
    
    for color in colors:
        bgr_color = hex_to_bgr(color['hex_color'])
        lower_bound = np.array([max(c-color['tolerance'], 0) for c in bgr_color])
        upper_bound = np.array([min(c+color['tolerance'], 255) for c in bgr_color])
        
        # Create a mask for the color range
        mask = cv2.inRange(image, lower_bound, upper_bound)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours around detected areas and record positions
        for contour in contours:
            min_area = color.get('min_area', 0)
            max_area = color.get('max_area', float('inf'))
            if min_area < cv2.contourArea(contour) < max_area:
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
                        walls.append(contour) # This adds walls
                    elif color['name'] == 'egg':
                        walls.append(contour)
                cv2.drawContours(image, [contour], -1, color['draw_color'], 2)

        update_grid_with_obstacles(image, grid, cell_height, cell_width)
    
    # Draw circles for detected ball and robot positions
    for pos in ball_positions:
        cv2.circle(image, pos, 5, (0, 0, 0), -1)  # Black circle for balls
    
    for pos in robot_positions:
        cv2.circle(image, pos, 5, (0, 0, 255), -1)  # Red circle for robots
    '''Darw the grid'''
    draw_grid(image, grid, cell_height, cell_width)

    if not ball_positions:
        print("No balls detected.")
    if not robot_positions:
        print("No robots detected.")
    if goal_position is None:
        print("No goal detected.")


    ball_positions = ball_positions[:10]
    robot_positions = robot_positions[:3]

    return ball_positions, robot_positions, goal_position, grid



# Define colors and their properties
colors = [
    {
        'name': 'balls',
        'hex_color': 'FDF7F5',
        'tolerance': 60,
        'min_area': 50,
        'max_area': 300,
        'draw_color': (0, 255, 0)  # Green
    },
    {
        'name': 'egg',
        'hex_color': 'FDF7F5',
        'tolerance': 20,
        'min_area' : 300,
        'max_area': 1000,
        'draw_color': (0, 0, 255) 
    },
    {
        'name': 'wall',
        'hex_color': 'ff5c0d',
        'tolerance': 80,
        'min_area': 500,
        'draw_color': (255, 0, 255)  # Purple
    },
    {
        'name': 'robot',
        'hex_color': '9AD9BB',
        'tolerance': 45,
        'min_area': 400,
        'draw_color': (255, 0, 0)  # Blue
    },
    {
        'name': 'goal',
        'hex_color': '427092',
        'tolerance': 20,
        'min_area': 50,
        'max_area': 500,
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

def render():
    reset()  # Assuming reset() is defined elsewhere
    ret, image = cam.read()  # Reading Images
    if not ret:
        print("Error: Failed to read frame.")
        return None

    # Detect multiple colors in the image
    ball_positions, robot_positions, goal_position, grid = detect_multiple_colors_in_image(image, colors)




    if len(robot_positions) != 3:
        print("Are we here")
        return None

    

    state = State(
        balls=[Ball(x, y, True) for x, y in ball_positions],
        corners=[],  # Update this if you need corners
        robot=Robot(*robot_positions[:3]),
        small_goal_pos=None,  # Update this if you have small_goal_pos
        big_goal_pos=goal_position,  # Update this if you have big_goal_pos
        grid=grid
    )


    
    if goal_position is not None:
        robot_center = robot_positions[0]  # Use the first robot position as the center
        distance_to_goal = calculate_distance(robot_center, goal_position)
        #(f"Distance to goal: {distance_to_goal}")

    # Display the frame with contours and circles
    cv2.imshow('Frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()



    return state


'''ALGORITHM WOOW'''


def plot_grid(grid, path, start, end):
    fig, ax = plt.subplots()
    rows, cols = len(grid), len(grid[0])

    # Create a color map for the grid
    cmap = plt.get_cmap('gray')
    cmap.set_under(color='black')

    # Plot the grid
    ax.imshow(grid, cmap=cmap, vmin=0.5)

    # Plot the start and end points
    ax.scatter(start[1], start[0], color='blue', s=100, label='Robot (Start)')
    ax.scatter(end[1], end[0], color='red', s=100, label='Ball (Goal)')

    # Plot the path
    if path:
        path_y, path_x = zip(*path)
        ax.plot(path_x, path_y, color='green', linewidth=2, marker='o', label='Path')

    # Add grid lines
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.grid(color='black', linestyle='-', linewidth=1)

    # Set axis labels and title
    ax.set_xticklabels(range(cols))
    ax.set_yticklabels(range(rows))
    ax.set_xlabel('Columns')
    ax.set_ylabel('Rows')
    ax.set_title('Robot Path Visualization')
    ax.legend()

    plt.gca().invert_yaxis()
    plt.show()

def test_algorithm():
    # Define the grid with 0s for empty spaces and 1s for walls
    grid = [
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

    # Define the robot's starting position and the ball's position
    start = (3, 5)  # (row, column)
    end = (0, 5)    # (row, column)

    # Run the A* algorithm to find the path
    path = a_star(grid, start, end)

    # Visualize the path on the grid
    plot_grid(grid, path, start, end)
    dest = find_straight_line_endpoint(path)
    print("First destination : ", dest)

def heuristic(a, b):
    # Using Manhattan distance as heuristic
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def is_valid_move(grid, current, neighbor):
    rows, cols = len(grid), len(grid[0])
    if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
        return False
    if grid[neighbor[0]][neighbor[1]] == 1:
        return False
    if abs(current[0] - neighbor[0]) == 1 and abs(current[1] - neighbor[1]) == 1: # Diagonal movement
        if grid[current[0]][neighbor[1]] == 1 or grid[neighbor[0]][current[1]] == 1:
            return False
    return True

def a_star(grid, start, end):
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

'''Check to see if it's a straight line'''
def find_straight_line_endpoint(path):
    if len(path) < 2:
        return None  # Path too short to determine direction changes

    # Initialize the direction vector
    direction = (path[1][0] - path[0][0], path[1][1] - path[0][1])

    for i in range(1, len(path) - 1):
        current_direction = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if current_direction != direction:
            return path[i]  # Return the position before the direction changes

    return path[-1]  # The whole path is straight
