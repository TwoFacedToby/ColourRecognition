import numpy as np
import cv2

# Convert hex color to BGR format
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

# Function to process video and draw contours around specific color
def detect_color_in_video(video_path, hex_color):
    # Convert hex color to BGR
    bgr_color = hex_to_bgr(hex_color)
    
    # Define the range of the color in BGR
    lower_bound = np.array([max(c-40, 0) for c in bgr_color])
    upper_bound = np.array([min(c+40, 255) for c in bgr_color])
    
    # Open video file
    cam = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        
        # Convert frame to BGR color space (by default it's already in BGR)
        bgr_frame = frame
        
        # Create a mask for the color range
        mask = cv2.inRange(bgr_frame, lower_bound, upper_bound)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours around detected areas
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter out small contours
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        
        # Display the frame with contours
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()

# Hex color to detect
hex_color = 'ff5c0d'

# Video path
video_path = "TrackVideos/Tester.mp4"

# Run the detection
detect_color_in_video(video_path, hex_color)
