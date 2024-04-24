import numpy as np
import cv2
from enum import Enum

# Capturing video through webcam
# cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture("TrackVideos/Position1_Shacky.mp4")


ballsState = []
wallsState = []
smallGoalPos = []
bigGoalPos = []
robotBack = []
robotFront = []


class Mask:
    def __init__(self, value, type):
        self.value = value
        self.type = type


class State:
    def __init__(self, balls, corners, robot, small_goal_pos, big_goal_pos):
        self.balls = balls
        self.corners = corners
        self.robot = robot
        self.small_goal_pos = small_goal_pos
        self.big_goal_pos = big_goal_pos


class Ball:
    def __init__(self, x, y, isOrange):
        self.x = x
        self.y = y
        self.isOrange = isOrange


class Robot:
    def __init__(self, front_x, front_y, back_x, back_y):
        self.front_x = front_x
        self.front_y = front_y
        self.back_x = back_x
        self.back_y = back_y


class Type(Enum):
    Wall = [0, 0, 0]
    Ball = [255, 255, 0]
    OrangeBall = [200, 255, 0]
    SmallBlueGoal = [255, 101, 0]
    BigGreenGoal = [30, 170, 70]
    Corner = [229, 178, 23]
    RobotBack = [50, 50, 50]
    RobotFront = [255, 255, 255]


class Color(Enum):
    WHITE = [np.array([0, 0, 200], np.uint8), np.array([180, 25, 255], np.uint8)]  # Range of White
    ORANGE = [np.array([10, 80, 180], np.uint8), np.array([14, 170, 200], np.uint8)]  # Range of Orange
    RED = [np.array([0, 50, 20], np.uint8), np.array([20, 100, 200], np.uint8)]  # Range of Red
    BROWN = [np.array([10, 30, 20], np.uint8), np.array([60, 100, 140], np.uint8)]  # Range of Brown
    GREEN = [np.array([25, 50, 70], np.uint8), np.array([50, 200, 200], np.uint8)]  # Range of Green
    BLUE = [np.array([100, 100, 100], np.uint8), np.array([130, 200, 200], np.uint8)]


def recognise_state_and_draw(image, mask):
    contours, hierarchy = cv2.findContours(mask.value, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    color = mask.type
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        show = "none"

        match type:
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
                    ballsState.append(Ball(x=x, y=y, isOrange=False))
                    show = "circle"

            case Type.OrangeBall:
                if 120 < area < 1500:
                    ballsState.append(Ball(x=x, y=y, isOrange=True))
                    show = "circle"

            case Type.RobotBack:
                if 120 < area < 1500:
                    robotBack.clear()
                    robotBack.append([x, y])
                    show = "square"
            case Type.RobotFront:
                if 120 < area < 1500:
                    robotFront.clear()
                    robotFront.append([x, y])
                    show = "square"
        if show == "circle":
            cv2.circle(image, (x, y), 15, (color[0], color[1], color[2]), 2)
        elif show == "square":
            cv2.rectangle(image, (x, y), (x + w, y + h), (color[0], color[1], color[2]), 2)


def mask_from_color(color, image_hsv, clean_image):
    return cv2.dilate(cv2.inRange(image_hsv, color[0], color[1]), clean_image)


def get_masks(image_hsv):
    clean_image = np.ones((5, 5), "uint8")
    masks = [Mask(value=mask_from_color(Color.WHITE.value, image_hsv, clean_image), type=Type.Ball),
             Mask(value=mask_from_color(Color.ORANGE.value, image_hsv, clean_image), type=Type.OrangeBall),
             Mask(value=mask_from_color(Color.RED.value, image_hsv, clean_image), type=Type.Wall),
             Mask(value=mask_from_color(Color.BROWN.value, image_hsv, clean_image), type=Type.Wall),
             Mask(value=mask_from_color(Color.GREEN.value, image_hsv, clean_image), type=Type.BigGreenGoal),
             Mask(value=mask_from_color(Color.BLUE.value, image_hsv, clean_image), type=Type.SmallBlueGoal)]
    return masks


def render():
    _, image = cam.read()  # Reading Images
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert the imageFrame in BGR to HSV
    masks = get_masks(image_hsv)
    ballsState.clear()

    for i in range(len(masks)):
        recognise_state_and_draw(image, masks[i])

    robot_location = Robot(front_x=600, front_y=200, back_x=600, back_y=300)  # TODO - move this to recognise_state
    cv2.rectangle(image, (600, 250), (600 + 50, 250 + 50),
                  (Type.RobotFront.value[0], Type.RobotFront.value[1], Type.RobotFront.value[2]), 2)

    ## End program
    cv2.imshow("Multiple Color Detection in Real-TIme", image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        # cap.release()
        cv2.destroyAllWindows()

    return State(
        balls=ballsState,
        corners=wallsState,
        robot=robot_location,
        small_goal_pos=smallGoalPos,
        big_goal_pos=bigGoalPos)
