import numpy as np
import cv2

# Hex to brg converter
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))


def detect_color_in_video(video_path, hex_color):
    
    bgr_color = hex_to_bgr(hex_color)
    
    # Define the range of the color in BGR (-80 +80 seems to be the best for this video with this color)
    lower_bound = np.array([max(c-80, 0) for c in bgr_color])
    upper_bound = np.array([min(c+80, 255) for c in bgr_color])
    
    
    cam = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        
        
        bgr_frame = frame
        
        # Create a mask for the color range
        mask = cv2.inRange(bgr_frame, lower_bound, upper_bound)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours around detected areas
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter out small contours!
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        
        # Display the frame with contours
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()

# Hex color to detect (this is color found with drop tool)
hex_color = 'ff5c0d'

video_path = "TrackVideos/Tester.mp4"

detect_color_in_video(video_path, hex_color)
