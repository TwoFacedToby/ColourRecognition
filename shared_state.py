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
low_x = (0, 0)
high_x = (0, 0)
real_position_robo = (1, 1)
half_field_pixel = 1

image = None



## Grid things
current_grid = None
current_cell_height = None
current_cell_width = None

## Wall values
lower_wall = None
upper_wall = None
right_wall = None
left_wall = None


## Cross-values
cross_middle = None
cross_top_left = None
cross_top_right = None
cross_bottom_left = None
cross_bottom_right = None

## temp orange
orange_ball = None

##
the_goal_pos = None