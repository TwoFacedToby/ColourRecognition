import cv2 as cv
import numpy as np

TrackbarWindow = "Trackbars"
CameraWindow = "Camera"

def nothing(x):
    pass

# Create a window for the trackbars
cv.namedWindow(TrackbarWindow, cv.WINDOW_NORMAL)
cv.resizeWindow(TrackbarWindow, 600, 200)  # Set window size for trackbars

# Create a window for the camera feed
cv.namedWindow(CameraWindow, cv.WINDOW_NORMAL)
cv.resizeWindow(CameraWindow, 800, 600)  # Set window size for camera feed

# B, G, R are for Lower Boundaries
# B2, G2, R2 are for Upper Boundaries
cv.createTrackbar('B', TrackbarWindow, 0, 255, nothing)
cv.createTrackbar('G', TrackbarWindow, 0, 255, nothing)
cv.createTrackbar('R', TrackbarWindow, 0, 255, nothing)
cv.createTrackbar('B2', TrackbarWindow, 0, 255, nothing)
cv.createTrackbar('G2', TrackbarWindow, 0, 255, nothing)
cv.createTrackbar('R2', TrackbarWindow, 0, 255, nothing)

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus

while cap.isOpened():
    _, frame = cap.read()
    B = cv.getTrackbarPos('B', TrackbarWindow)
    G = cv.getTrackbarPos('G', TrackbarWindow)
    R = cv.getTrackbarPos('R', TrackbarWindow)
    B2 = cv.getTrackbarPos('B2', TrackbarWindow)
    G2 = cv.getTrackbarPos('G2', TrackbarWindow)
    R2 = cv.getTrackbarPos('R2', TrackbarWindow)
    lower_boundary = np.array([B, G, R])
    upper_boundary = np.array([B2, G2, R2])
    mask = cv.inRange(frame, lower_boundary, upper_boundary)
    final = cv.bitwise_and(frame, frame, mask=mask)
    cv.imshow(CameraWindow, final)

    if cv.waitKey(1) == ord('q'): break

cap.release()
cv.destroyAllWindows()
