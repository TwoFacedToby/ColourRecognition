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

# H, S, V are for Lower Boundaries
# H2, S2, V2 are for Upper Boundaries
cv.createTrackbar('H', TrackbarWindow, 0, 255, nothing)
cv.createTrackbar('S', TrackbarWindow, 0, 255, nothing)
cv.createTrackbar('V', TrackbarWindow, 0, 255, nothing)
cv.createTrackbar('H2', TrackbarWindow, 0, 255, nothing)
cv.createTrackbar('S2', TrackbarWindow, 0, 255, nothing)
cv.createTrackbar('V2', TrackbarWindow, 0, 255, nothing)

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus

while cap.isOpened():
    _, frame = cap.read()
    H = cv.getTrackbarPos('H', TrackbarWindow)
    S = cv.getTrackbarPos('S', TrackbarWindow)
    V = cv.getTrackbarPos('V', TrackbarWindow)
    H2 = cv.getTrackbarPos('H2', TrackbarWindow)
    S2 = cv.getTrackbarPos('S2', TrackbarWindow)
    V2 = cv.getTrackbarPos('V2', TrackbarWindow)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower_boundary = np.array([H, S, V])
    upper_boundary = np.array([H2, S2, V2])
    mask = cv.inRange(hsv, lower_boundary, upper_boundary)
    final = cv.bitwise_and(frame, frame, mask=mask)
    cv.imshow(CameraWindow, final)

    if cv.waitKey(1) == ord('q'): break

cap.release()
cv.destroyAllWindows()
