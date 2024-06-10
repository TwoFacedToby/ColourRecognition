import numpy as np
import cv2
#cv2.CAP_DSHOW

# Hex to BGR converter
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

def detect_multiple_colors_in_video(video_path, colors):
    # Open the video capture
    print("Opening video capture...")
    cam = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)

    

    
    if not cam.isOpened():
        print("Error: Could not open video capture.")
        return
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Failed to read frame.")
            break


        ball_positions = []
        robot_positions = []
        
        for color in colors:
            bgr_color = hex_to_bgr(color['hex_color'])
            lower_bound = np.array([max(c-color['tolerance'], 0) for c in bgr_color])
            upper_bound = np.array([min(c+color['tolerance'], 255) for c in bgr_color])
            
            # Create a mask for the color range
            mask = cv2.inRange(frame, lower_bound, upper_bound)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours around detected areas
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
                    cv2.drawContours(frame, [contour], -1, color['draw_color'], 2)
        

        # Draw circles for detected ball and robot positions
        for pos in ball_positions:
            cv2.circle(frame, pos, 5, (0, 0, 0), -1)  # Black circle for balls
        
        for pos in robot_positions:
            cv2.circle(frame, pos, 5, (0, 0, 255), -1)  # Red circle for robots


        # Display the frame with contours
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()

    # Limit the number of positions to 10 balls and 3 robot positions
    ball_positions = ball_positions[:10]
    robot_positions = robot_positions[:3]
    
    print("Ball Positions:", ball_positions)
    print("Robot Positions:", robot_positions)

# Define colors and their properties
colors = [
    {
        'name': 'balls',
        'hex_color': 'FDF7F5',
        'tolerance': 80,
        'min_area': 50,
        'max_area': 300,
        'draw_color': (0, 255, 0)  # Green
    },
    {
        'name': 'egg',
        'hex_color': 'FDF7F5',
        'tolerance': 80,
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
        'tolerance': 40,
        'min_area': 500,
        'draw_color': (255, 0, 0)  # Blue
    }
    
]

# Use the default webcam (index 0)
video_path = 0

detect_multiple_colors_in_video(video_path, colors)





'''
Settings for walls:
    hex_color = 'ff5c0d'
    lower_bound = np.array([max(c-80, 0) for c in bgr_color])
    upper_bound = np.array([min(c+80, 255) for c in bgr_color])
    for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter out small contours!
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

Settings for robot black spots
    hex_color = '2F2E2E'
    lower_bound = np.array([max(c-40, 0) for c in bgr_color])
    upper_bound = np.array([min(c+40, 255) for c in bgr_color])
    for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter out small contours!
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

Settings for white balls:


'''
