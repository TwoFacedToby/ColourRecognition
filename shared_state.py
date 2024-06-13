class Ball:
    def __init__(self, x, y, is_orange):
        self.x = x
        self.y = y
        self.isOrange = is_orange

# shared_state.py
reference_vector_magnitude = 1  # Default value to prevent division by zero
real_world_distance = 1150  # Real-world distance between the highest and lowest x-points in runtime units
image_height = 1
current_ball = None
middlepoint = None
low_x = 1

real_position_robo = None