import numpy as np
import cv2
from enum import Enum
from collections import Counter, deque
import shared_state
import math
import tkinter as tk
from tkinter import ttk
import threading

use_video = False

if use_video: # Video
    cam = cv2.VideoCapture("TrackVideos/new_room_new_video.mp4")
else: # Camera
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus

def get_colors():
    return [
        {
            'name': 'balls',
            'hex_color': 'FEFDFD',
            'h_lower': 0,
            'h_upper': 255,
            's_lower': 0,
            's_upper': 50,
            'v_lower': 210,
            'v_upper': 255,
            'min_area': 100,
            'max_area': 200,
            'draw_color': (0, 100, 255)
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
            's_lower': 104,
            's_upper': 255,
            'v_lower': 141,
            'v_upper': 255,
            'min_area': 400,
            'max_area': 99999999,
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
            's_upper': 255,
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



colors = get_colors()

tolerance_entry = None
min_area_entry = None
max_area_entry = None
tolerance_trace_id = None
min_area_trace_id = None
max_area_trace_id = None

# Global variable to store the picked color
picked_color = None

def pick_color(event, x, y, flags, param):
    global picked_color
    if event == cv2.EVENT_LBUTTONDOWN:
        b, g, r = param[y, x]
        picked_color = '{:02x}{:02x}{:02x}'.format(r, g, b).upper()
        print(f"Picked color: {picked_color}")  # Debug statement
        hex_color_var.set(picked_color)  # Update the hex color variable

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

def hex_to_hsv(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return 0, 0, 0  # Return black if the hex code is incomplete
    bgr = tuple(int(hex_color[i:i + 2], 16) for i in (4, 2, 0))
    bgr = np.uint8([[bgr]])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return hsv[0][0]

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
                # Calculate aspect ratio and extent
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                rect_area = w * h
                extent = area / float(rect_area)

                hu_moment1_lower = 0.15  # Lower bound of the range
                hu_moment1_upper = 0.18  # Upper bound of the range
                # Define the range for circularity
                circularity_lower = 0.8  # Lower bound of the circularity range
                circularity_upper = 1.2  # Upper bound of the circularity range

                perimeter = cv2.arcLength(contour, True)
                # Calculate circularity
                circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

                # Define thresholds for filtering
                aspect_ratio_threshold = 0.5  # Broadened threshold for aspect ratio
                extent_threshold = 0.5  # Lowered threshold for extent

                # Calculate Hu Moments
                hu_moments = cv2.HuMoments(cv2.moments(contour)).flatten()
                hu_moment1 = hu_moments[0]


                if color['name'] == 'robot':
                    if not (aspect_ratio_threshold <= aspect_ratio <= 1 / aspect_ratio_threshold and
                    extent > extent_threshold and
                    hu_moment1_lower <= hu_moment1 <= hu_moment1_upper):
                        continue  # Skip contours that are not close to rectangular

                
                



                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cX = int(M['m10'] / M['m00'])
                    cY = int(M['m01'] / M['m00'])
                    if color['name'] == 'balls':
                        
                        if circularity_lower <= circularity <= circularity_upper:
                            print(f"Ball Contour, Circularity: {circularity}")
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

    #for pos in ball_positions:
        #cv2.circle(image, pos, 5, (0, 0, 0), -1)

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



color_picker_enabled = False

def render():
    global color_picker_enabled
    ret, image = cam.read()  # Reading Images
    if not ret:
        print("Error: Failed to read frame.")
        cam.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart the video
        return None

    if use_video:
        screen_width = 1920  # Change this to your screen width
        screen_height = 1080  # Change this to your screen height
        new_width = screen_width // 2
        new_height = screen_height // 2
        image = cv2.resize(image, (new_width, new_height))

    detect_multiple_colors_in_image(image, colors)

    if color_picker_enabled:
        cv2.setMouseCallback('Frame', pick_color, param=image)
        color_picker_enabled = False

    cv2.imshow('Frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

def remove_traces():
    h_lower_var.trace_remove('write', h_lower_trace_id)
    h_upper_var.trace_remove('write', h_upper_trace_id)
    s_lower_var.trace_remove('write', s_lower_trace_id)
    s_upper_var.trace_remove('write', s_upper_trace_id)
    v_lower_var.trace_remove('write', v_lower_trace_id)
    v_upper_var.trace_remove('write', v_upper_trace_id)
    min_area_entry_var.trace_remove('write', min_area_trace_id)
    max_area_entry_var.trace_remove('write', max_area_trace_id)

def add_traces():
    global h_lower_trace_id, h_upper_trace_id, s_lower_trace_id, s_upper_trace_id, v_lower_trace_id, v_upper_trace_id
    global min_area_trace_id, max_area_trace_id
    h_lower_trace_id = h_lower_var.trace_add('write', update_params_from_entries)
    h_upper_trace_id = h_upper_var.trace_add('write', update_params_from_entries)
    s_lower_trace_id = s_lower_var.trace_add('write', update_params_from_entries)
    s_upper_trace_id = s_upper_var.trace_add('write', update_params_from_entries)
    v_lower_trace_id = v_lower_var.trace_add('write', update_params_from_entries)
    v_upper_trace_id = v_upper_var.trace_add('write', update_params_from_entries)
    min_area_trace_id = min_area_entry_var.trace_add('write', update_params_from_entries)
    max_area_trace_id = max_area_entry_var.trace_add('write', update_params_from_entries)

def update_color_params_without_traces():
    selected_color = color_var.get()
    for color in colors:
        if color['name'] == selected_color:
            hex_color_var.set(color.get('hex_color', '#000000'))
            h_lower_var.set(str(color.get('h_lower', 0)))
            h_upper_var.set(str(color.get('h_upper', 0)))
            s_lower_var.set(str(color.get('s_lower', 0)))
            s_upper_var.set(str(color.get('s_upper', 0)))
            v_lower_var.set(str(color.get('v_lower', 0)))
            v_upper_var.set(str(color.get('v_upper', 0)))
            min_area_entry_var.set(str(color.get('min_area', 0)))
            max_area_entry_var.set(str(color.get('max_area', 0)))
            h_lower_entry.delete(0, tk.END)
            h_lower_entry.insert(0, str(color.get('h_lower', 0)))
            h_upper_entry.delete(0, tk.END)
            h_upper_entry.insert(0, str(color.get('h_upper', 0)))
            s_lower_entry.delete(0, tk.END)
            s_lower_entry.insert(0, str(color.get('s_lower', 0)))
            s_upper_entry.delete(0, tk.END)
            s_upper_entry.insert(0, str(color.get('s_upper', 0)))
            v_lower_entry.delete(0, tk.END)
            v_lower_entry.insert(0, str(color.get('v_lower', 0)))
            v_upper_entry.delete(0, tk.END)
            v_upper_entry.insert(0, str(color.get('v_upper', 0)))
            min_area_entry.delete(0, tk.END)
            min_area_entry.insert(0, str(color.get('min_area', 0)))
            max_area_entry.delete(0, tk.END)
            max_area_entry.insert(0, str(color.get('max_area', 0)))
            break

def update_color_params(*args):
    selected_color = color_var.get()
    for color in colors:
        if color['name'] == selected_color:
            print(f"Updating params for {selected_color}: {color}")
            remove_traces()
            hex_color_var.set(color.get('hex_color', '#000000'))
            h_lower_var.set(str(color.get('h_lower', 0)))
            h_upper_var.set(str(color.get('h_upper', 0)))
            s_lower_var.set(str(color.get('s_lower', 0)))
            s_upper_var.set(str(color.get('s_upper', 0)))
            v_lower_var.set(str(color.get('v_lower', 0)))
            v_upper_var.set(str(color.get('v_upper', 0)))
            min_area_entry_var.set(str(color.get('min_area', 0)))
            max_area_entry_var.set(str(color.get('max_area', 0)))
            h_lower_entry.delete(0, tk.END)
            h_lower_entry.insert(0, str(color.get('h_lower', 0)))
            h_upper_entry.delete(0, tk.END)
            h_upper_entry.insert(0, str(color.get('h_upper', 0)))
            s_lower_entry.delete(0, tk.END)
            s_lower_entry.insert(0, str(color.get('s_lower', 0)))
            s_upper_entry.delete(0, tk.END)
            s_upper_entry.insert(0, str(color.get('s_upper', 0)))
            v_lower_entry.delete(0, tk.END)
            v_lower_entry.insert(0, str(color.get('v_lower', 0)))
            v_upper_entry.delete(0, tk.END)
            v_upper_entry.insert(0, str(color.get('v_upper', 0)))
            min_area_entry.delete(0, tk.END)
            min_area_entry.insert(0, str(color.get('min_area', 0)))
            max_area_entry.delete(0, tk.END)
            max_area_entry.insert(0, str(color.get('max_area', 0)))
            add_traces()
            break

def update_params_from_entries(*args):
    try:
        h_lower = int(h_lower_var.get())
        h_upper = int(h_upper_var.get())
        s_lower = int(s_lower_var.get())
        s_upper = int(s_upper_var.get())
        v_lower = int(v_lower_var.get())
        v_upper = int(v_upper_var.get())
        min_area = int(min_area_entry_var.get())
        max_area = int(max_area_entry_var.get())
        hex_color = hex_color_var.get()
    except ValueError:
        return
    selected_color = color_var.get()
    for color in colors:
        if color['name'] == selected_color:
            print(f"Updating color settings from entries for {selected_color}")
            color['h_lower'] = h_lower
            color['h_upper'] = h_upper
            color['s_lower'] = s_lower
            color['s_upper'] = s_upper
            color['v_lower'] = v_lower
            color['v_upper'] = v_upper
            color['min_area'] = min_area
            color['max_area'] = max_area
            color['hex_color'] = hex_color
            break


def create_gui():
    global color_var, hex_color_var, h_lower_var, h_upper_var, s_lower_var, s_upper_var, v_lower_var, v_upper_var
    global min_area_entry_var, max_area_entry_var
    global h_lower_entry, h_upper_entry, s_lower_entry, s_upper_entry, v_lower_entry, v_upper_entry
    global min_area_entry, max_area_entry
    global h_lower_trace_id, h_upper_trace_id, s_lower_trace_id, s_upper_trace_id, v_lower_trace_id, v_upper_trace_id
    global min_area_trace_id, max_area_trace_id

    def enable_color_picker():
        global color_picker_enabled
        color_picker_enabled = True

    root = tk.Tk()
    root.title("Color Picker")

    color_var = tk.StringVar()
    hex_color_var = tk.StringVar()
    h_lower_var = tk.StringVar()
    h_upper_var = tk.StringVar()
    s_lower_var = tk.StringVar()
    s_upper_var = tk.StringVar()
    v_lower_var = tk.StringVar()
    v_upper_var = tk.StringVar()
    min_area_entry_var = tk.StringVar()
    max_area_entry_var = tk.StringVar()

    color_var.set(colors[0]['name'])

    color_menu = ttk.Combobox(root, textvariable=color_var)
    color_menu['values'] = [color['name'] for color in colors]
    color_menu.grid(row=0, column=0, padx=10, pady=10)
    color_menu.bind("<<ComboboxSelected>>", update_color_params)

    tk.Label(root, text="Hex Color:").grid(row=1, column=0, padx=10, pady=10)
    hex_color_entry = tk.Entry(root, textvariable=hex_color_var)
    hex_color_entry.grid(row=1, column=1, padx=10, pady=10)

    tk.Label(root, text="Hue Lower:").grid(row=2, column=0, padx=10, pady=10)
    h_lower_entry = tk.Entry(root, textvariable=h_lower_var)
    h_lower_entry.grid(row=2, column=1, padx=10, pady=10)

    tk.Label(root, text="Hue Upper:").grid(row=3, column=0, padx=10, pady=10)
    h_upper_entry = tk.Entry(root, textvariable=h_upper_var)
    h_upper_entry.grid(row=3, column=1, padx=10, pady=10)

    tk.Label(root, text="Saturation Lower:").grid(row=4, column=0, padx=10, pady=10)
    s_lower_entry = tk.Entry(root, textvariable=s_lower_var)
    s_lower_entry.grid(row=4, column=1, padx=10, pady=10)

    tk.Label(root, text="Saturation Upper:").grid(row=5, column=0, padx=10, pady=10)
    s_upper_entry = tk.Entry(root, textvariable=s_upper_var)
    s_upper_entry.grid(row=5, column=1, padx=10, pady=10)

    tk.Label(root, text="Value Lower:").grid(row=6, column=0, padx=10, pady=10)
    v_lower_entry = tk.Entry(root, textvariable=v_lower_var)
    v_lower_entry.grid(row=6, column=1, padx=10, pady=10)

    tk.Label(root, text="Value Upper:").grid(row=7, column=0, padx=10, pady=10)
    v_upper_entry = tk.Entry(root, textvariable=v_upper_var)
    v_upper_entry.grid(row=7, column=1, padx=10, pady=10)

    tk.Label(root, text="Min Area:").grid(row=8, column=0, padx=10, pady=10)
    min_area_entry = tk.Entry(root, textvariable=min_area_entry_var)
    min_area_entry.grid(row=8, column=1, padx=10, pady=10)

    tk.Label(root, text="Max Area:").grid(row=9, column=0, padx=10, pady=10)
    max_area_entry = tk.Entry(root, textvariable=max_area_entry_var)
    max_area_entry.grid(row=9, column=1, padx=10, pady=10)

    color_picker_button = tk.Button(root, text="Pick Color from Image", command=enable_color_picker)
    color_picker_button.grid(row=10, column=0, columnspan=2, padx=10, pady=10)

    update_color_params_without_traces()

    add_traces()

    root.mainloop()



def adjust_exposure(image):
    alpha = 1
    beta = -50

    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def normalize_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    normalized_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return normalized_image

if __name__ == "__main__":
    gui_thread = threading.Thread(target=create_gui)
    gui_thread.start()

    while True:
        render()
