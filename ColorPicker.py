import numpy as np
import cv2
from enum import Enum
from collections import Counter, deque
import shared_state
import math
import tkinter as tk
from tkinter import ttk
import threading

# Camera
# cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus

# Video
cam = cv2.VideoCapture("TrackVideos/new_room_new_video.mp4")

colors = [
    {
        'name': 'balls',
        'hex_color': 'FDF7F5',
        'tolerance': 80,
        'min_area': 50,
        'max_area': 200,
        'draw_color': (0, 255, 0)  # Green
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
        'min_area': 500,
        'max_area': 0,
        'draw_color': (255, 0, 255)  # Purple
    },
    {
        'name': 'robot',
        'hex_color': '9AD9BB',
        'tolerance': 45,
        'min_area': 400,
        'max_area': 800,
        'draw_color': (255, 0, 0)  # Blue
    },
    {
        'name': 'goal',
        'hex_color': 'ADA0BD',
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


tolerance_entry = None
min_area_entry = None
max_area_entry = None
tolerance_trace_id = None
min_area_trace_id = None
max_area_trace_id = None



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


def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return 0, 0, 0  # Return black if the hex code is incomplete
    return tuple(int(hex_color[i:i + 2], 16) for i in (4, 2, 0))


def vector_between_points(point1, point2):
    """ Calculate the vector between two points (x, y). """
    return np.array([point1[0] - point2[0], point1[1] - point2[1]])


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
                        # Append all x-values from the contour to wall_x_positions
                        for point in contour:
                            x, y = point[0]
                            if middle_x - 80 <= x <= middle_x + 80 and middle_y - 80 <= y <= middle_y + 80:
                                cross_positions.append((x, y))
                            else:
                                wall_x_positions.append(x)
                                wall_y_positions.append(y)
                cv2.drawContours(image, [contour], -1, color['draw_color'], 2)

    # Draw circles for detected ball and robot positions
    for pos in ball_positions:
        cv2.circle(image, pos, 5, (0, 0, 0), -1)  # Black circle for balls

    for pos in robot_positions:
        cv2.circle(image, pos, 5, (0, 0, 255), -1)  # Red circle for robots
    ##
    #if not ball_positions:
    # print("No balls detected.")
    #if not robot_positions:
    # print("No robots detected.")
    #if goal_position is None:
    # print("No goal detected.")

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
        cv2.circle(image, highest_x_with_center_y, 5, (0, 0, 0),
                   -1)  # Black dot for the most common high x with center y

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
        cv2.circle(image, highest_y_with_center_x, 5, (0, 0, 0),
                   -1)  # Black dot for the most common high y with center x

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

        cv2.circle(image, middle_cross_point, 5, (0, 255, 0), -1)  # Only draw if middle_cross_point is assigned

    ball_positions = ball_positions[:10]
    robot_positions = robot_positions[:3]

    # Draw a circle at the exact middle of the image
    cv2.circle(image, (middle_x, middle_y), 5, (0, 255, 0), -1)  # Green circle for the middle point

    return ball_positions, robot_positions, goal_position


def render():
    ret, image = cam.read()  # Reading Images
    if not ret:
        print("Error: Failed to read frame.")
        cam.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart the video
        return None

    # Resize the image to 1/4th of the screen size
    screen_width = 1920  # Change this to your screen width
    screen_height = 1080  # Change this to your screen height
    new_width = screen_width // 2
    new_height = screen_height // 2
    image = cv2.resize(image, (new_width, new_height))

    detect_multiple_colors_in_image(image, colors)

    cv2.imshow('Frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
def remove_traces():
    tolerance_entry_var.trace_remove('write', tolerance_trace_id)
    min_area_entry_var.trace_remove('write', min_area_trace_id)
    max_area_entry_var.trace_remove('write', max_area_trace_id)

def add_traces():
    global tolerance_trace_id, min_area_trace_id, max_area_trace_id
    tolerance_trace_id = tolerance_entry_var.trace_add('write', update_params_from_entries)
    min_area_trace_id = min_area_entry_var.trace_add('write', update_params_from_entries)
    max_area_trace_id = max_area_entry_var.trace_add('write', update_params_from_entries)
def update_color_params_without_traces():
    selected_color = color_var.get()
    for color in colors:
        if color['name'] == selected_color:
            hex_color_var.set(color.get('hex_color', '#000000'))  # Ensure the hex color is updated
            tolerance_entry_var.set(str(color.get('tolerance', 0)))
            min_area_entry_var.set(str(color.get('min_area', 0)))
            max_area_entry_var.set(str(color.get('max_area', 0)))

            # Force entry widgets to update their displayed values
            tolerance_entry.delete(0, tk.END)
            tolerance_entry.insert(0, str(color.get('tolerance', 0)))

            min_area_entry.delete(0, tk.END)
            min_area_entry.insert(0, str(color.get('min_area', 0)))

            max_area_entry.delete(0, tk.END)
            max_area_entry.insert(0, str(color.get('max_area', 0)))
            break


def update_color_params(*args):
    selected_color = color_var.get()
    for color in colors:
        if color['name'] == selected_color:
            print(f"Updating params for {selected_color}: {color}")  # Debug statement

            remove_traces()  # Temporarily remove traces

            hex_color_var.set(color.get('hex_color', '#000000'))  # Ensure the hex color is updated
            tolerance_entry_var.set(str(color.get('tolerance', 0)))
            min_area_entry_var.set(str(color.get('min_area', 0)))
            max_area_entry_var.set(str(color.get('max_area', 0)))

            # Force entry widgets to update their displayed values
            tolerance_entry.delete(0, tk.END)
            tolerance_entry.insert(0, str(color.get('tolerance', 0)))

            min_area_entry.delete(0, tk.END)
            min_area_entry.insert(0, str(color.get('min_area', 0)))

            max_area_entry.delete(0, tk.END)
            max_area_entry.insert(0, str(color.get('max_area', 0)))

            add_traces()  # Re-add traces
            break


def update_params_from_entries(*args):
    try:
        tolerance = int(tolerance_entry_var.get())
        min_area = int(min_area_entry_var.get())
        max_area = int(max_area_entry_var.get())
        hex_color = hex_color_var.get()  # Ensure the hex color is read
    except ValueError:
        return  # Ignore if the entry is not a valid integer or hex color
    selected_color = color_var.get()
    for color in colors:
        if color['name'] == selected_color:
            print(f"Updating color settings from entries for {selected_color}")  # Debug statement
            color['tolerance'] = tolerance
            color['min_area'] = min_area
            color['max_area'] = max_area
            color['hex_color'] = hex_color  # Ensure the hex color is updated
            break


def create_gui():
    global color_var, hex_color_var, tolerance_entry_var, min_area_entry_var, max_area_entry_var
    global tolerance_entry, min_area_entry, max_area_entry  # Make these accessible in update_color_params
    global tolerance_trace_id, min_area_trace_id, max_area_trace_id  # Add these global variables

    root = tk.Tk()
    root.title("Color Picker")

    color_var = tk.StringVar()
    hex_color_var = tk.StringVar()
    tolerance_entry_var = tk.StringVar()
    min_area_entry_var = tk.StringVar()
    max_area_entry_var = tk.StringVar()

    color_var.set(colors[0]['name'])  # Set default value

    color_menu = ttk.Combobox(root, textvariable=color_var)
    color_menu['values'] = [color['name'] for color in colors]
    color_menu.grid(row=0, column=0, padx=10, pady=10)
    color_menu.bind("<<ComboboxSelected>>", update_color_params)

    tk.Label(root, text="Hex Color:").grid(row=1, column=0, padx=10, pady=10)
    hex_color_entry = tk.Entry(root, textvariable=hex_color_var)
    hex_color_entry.grid(row=1, column=1, padx=10, pady=10)

    tk.Label(root, text="Tolerance:").grid(row=2, column=0, padx=10, pady=10)
    tolerance_entry = tk.Entry(root, textvariable=tolerance_entry_var)
    tolerance_entry.grid(row=2, column=1, padx=10, pady=10)

    tk.Label(root, text="Min Area:").grid(row=3, column=0, padx=10, pady=10)
    min_area_entry = tk.Entry(root, textvariable=min_area_entry_var)
    min_area_entry.grid(row=3, column=1, padx=10, pady=10)

    tk.Label(root, text="Max Area:").grid(row=4, column=0, padx=10, pady=10)
    max_area_entry = tk.Entry(root, textvariable=max_area_entry_var)
    max_area_entry.grid(row=4, column=1, padx=10, pady=10)

    # Set initial values for entries without traces
    update_color_params_without_traces()

    # Initialize traces
    add_traces()

    root.mainloop()



if __name__ == "__main__":
    gui_thread = threading.Thread(target=create_gui)
    gui_thread.start()

    while True:
        render()
