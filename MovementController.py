import math

import numpy as np


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
    return np.array([robot.x - ball.x, robot.y - ball.y])
