import numpy as np
import cv2
from enum import Enum

# Capturing video through webcam
# cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture("TrackVideos/Position1_normal.mp4")


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
    Ball = [255, 255, 0]
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
    Yellow = 6


ballsState = []
wallsState = []
smallGoalPos = []
bigGoalPos = []
robot_state = Robot([0, 0], [0, 0], [0, 0])
walls = []


def obstacle_lines(image):
    # convert to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # perform edge detection
    edges = cv2.Canny(grayscale, 20, 100)
    # detect lines in the image using hough lines technique
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60, np.array([]), 50, 5)
    # iterate over the output lines and draw them
    if len(walls) < 100:
        walls.append(lines)
    else:
        walls.pop(0)
        walls.append(lines)
    for wall in walls:
        for line in wall:
            for x1, y1, x2, y2 in line:
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.line(edges, (x1, y1), (x2, y2), (255, 0, 0), 2)


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
                if 100 < area < 1500:
                    bigGoalPos.clear()
                    bigGoalPos.append([x, y])
                    show = "square"

            case Type.Ball:
                if 120 < area < 1500:
                    ballsState.append(Ball(x=x, y=y, is_orange=False))
                    show = "circle"

            case Type.OrangeBall:
                if 120 < area < 1500:
                    ballsState.append(Ball(x=x, y=y, is_orange=True))
                    show = "circle"

            case Type.Robot:
                if 120 < area < 1500:
                    if robot_state.pos_1 == [0, 0]:
                        robot_state.pos_1 = [x, y]
                    elif robot_state.pos_2 == [0, 0]:
                        robot_state.pos_2 = [x, y]
                    elif robot_state.pos_3 == [0, 0]:
                        robot_state.pos_3 = [x, y]
                    show = "square"
        if show == "circle":
            cv2.circle(image, (x + 15, y + 15), 15, (color[0], color[1], color[2]), 2)
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
            color_low = np.array([0, 50, 20], np.uint8)
            color_high = np.array([20, 100, 200], np.uint8)
        case Color.BROWN:
            color_low = np.array([10, 30, 20], np.uint8)
            color_high = np.array([60, 100, 140], np.uint8)
        case Color.GREEN:
            color_low = np.array([25, 50, 70], np.uint8)
            color_high = np.array([50, 200, 200], np.uint8)
        case Color.BLUE:
            color_low = np.array([100, 100, 100], np.uint8)
            color_high = np.array([130, 200, 200], np.uint8)
        case Color.Yellow:
            color_low = np.array([200, 177, 123], np.uint8)
            color_high = np.array([173, 146, 79], np.uint8)
    return cv2.dilate(cv2.inRange(image_hsv, color_low, color_high), clean_image)


def get_types():
    return [Type.Ball,
            Type.OrangeBall,
            Type.Wall,
            Type.Wall,
            Type.BigGreenGoal,
            Type.SmallBlueGoal,
            Type.Robot]


def get_masks(image_hsv):
    clean_image = np.ones((5, 5), "uint8")
    return [mask_from_color(Color.WHITE, image_hsv, clean_image),
            mask_from_color(Color.ORANGE, image_hsv, clean_image),
            mask_from_color(Color.RED, image_hsv, clean_image),
            mask_from_color(Color.BROWN, image_hsv, clean_image),
            mask_from_color(Color.GREEN, image_hsv, clean_image),
            mask_from_color(Color.BLUE, image_hsv, clean_image),
            mask_from_color(Color.Yellow, image_hsv, clean_image)]


def reset():
    ballsState.clear()
    global robot_state
    robot_state = Robot([0, 0], [0, 0], [0, 0])


def draw_robot(image):
    cv2.rectangle(image, (600, 250), (600 + 50, 250 + 50),
                  (Type.Robot.value[0], Type.Robot.value[1], Type.Robot.value[2]), 2)


def render():
    reset()
    _, image = cam.read()  # Reading Images
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert the imageFrame in BGR to HSV
    masks = get_masks(image_hsv)
    types = get_types()
    for i in range(len(masks)):
        recognise_state_and_draw(image, masks[i], types[i])
    obstacle_lines(image)
    draw_robot(image)

    cv2.imshow("Multiple Color Detection in Real-TIme", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    return State(
        balls=ballsState,
        corners=wallsState,
        robot=robot_state,
        small_goal_pos=smallGoalPos,
        big_goal_pos=bigGoalPos)
