import cv2
import numpy as np
from tkinter import *
from tkinter import ttk

class HSVColorFinder:
    def __init__(self, root):
        self.root = root
        self.root.title("HSV Color Finder")

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Camera Feed
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus

        # Desired width and height for the camera feed window
        self.window_width = 640
        self.window_height = 480

        self.lower_hue = IntVar(value=0)
        self.upper_hue = IntVar(value=255)
        self.lower_saturation = IntVar(value=0)
        self.upper_saturation = IntVar(value=255)
        self.lower_value = IntVar(value=0)
        self.upper_value = IntVar(value=255)

        self.create_ui()

        self.update()

    def create_ui(self):
        frame = Frame(self.root)
        frame.pack(side=LEFT, padx=10, pady=10)

        self.create_control(frame, "Lower Hue", self.lower_hue, 0, 255)
        self.create_control(frame, "Upper Hue", self.upper_hue, 0, 255)
        self.create_control(frame, "Lower Saturation", self.lower_saturation, 0, 255)
        self.create_control(frame, "Upper Saturation", self.upper_saturation, 0, 255)
        self.create_control(frame, "Lower Value", self.lower_value, 0, 255)
        self.create_control(frame, "Upper Value", self.upper_value, 0, 255)

    def create_control(self, parent, label, variable, min_val, max_val):
        label = Label(parent, text=label)
        label.pack(anchor=W)

        entry = Entry(parent, textvariable=variable)
        entry.pack(anchor=W)

        slider = Scale(parent, from_=min_val, to=max_val, orient=HORIZONTAL, variable=variable)
        slider.pack(anchor=W)

    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.quit()
            return

        # Resize the frame to the desired window size
        frame = cv2.resize(frame, (self.window_width, self.window_height))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        


        lower_bound = np.array([self.lower_hue.get(), self.lower_saturation.get(), self.lower_value.get()])
        upper_bound = np.array([self.upper_hue.get(), self.upper_saturation.get(), self.upper_value.get()])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        

        result = cv2.bitwise_and(frame, frame, mask=mask)

        

        cv2.imshow("Camera Feed", frame)
        cv2.imshow("Mask", result)

        self.root.after(10, self.update)

    def on_closing(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    root = Tk()
    app = HSVColorFinder(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()