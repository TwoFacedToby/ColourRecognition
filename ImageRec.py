import numpy as np 
import cv2 
  
  
## Capturing video through webcam 
#cam = cv2.VideoCapture(0) 
cam = cv2.VideoCapture("./TestVideo/Position1_normal.mp4")

## Making a box to show the colour/object detected
def boxColour(image, mask, colour, display):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
  
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 100 and area < 1500): 
            x, y, w, h = cv2.boundingRect(contour) 
            cv2.rectangle(image, (x, y), (x + w, y + h), (colour[0], colour[1], colour[2]), 2) 
            cv2.putText(image, display, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (colour[0], colour[1], colour[2]))
    
    #return image    
  
# We aint stopping this thing
while(1): 

    # Reading images
    _, image = cam.read() 

    # Convert the imageFrame in BGR to HSV
    HSVImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 

##Setting up the different colours/objects to recognize!

    #Setting up the two types of balls
  
    # Range of white/Ball1
    white_low = np.array([0, 0, 200], np.uint8) 
    white_high = np.array([180, 25, 255], np.uint8) 

    # What is white/Ball1
    white_mask = cv2.inRange(HSVImage, white_low, white_high) 

    # Range of orange/Ball2
    #orange_low = np.array([5, 100, 100], np.uint8) 
    #orange_high = np.array([15, 255, 255], np.uint8) 
    #orange_low = np.array([10, 200, 100], np.uint8) 
    
    orange_low = np.array([10, 80, 180], np.uint8)
    orange_high = np.array([14, 170, 200], np.uint8)
    # BROWN = orange_high = np.array([25, 100, 160], np.uint8) or  orange_high = np.array([15, 170, 190], np.uint8)

    # What is orange/Ball2
    orange_mask = cv2.inRange(HSVImage, orange_low, orange_high) 


    #Setting up obstacles
    
    # Range of red/obstacle
    #red_low = np.array([136, 87, 111], np.uint8) 
    #red_high = np.array([180, 255, 255], np.uint8) 
    red_low = np.array([150, 100, 80], np.uint8) 
    red_high = np.array([200, 255, 255], np.uint8) 

    # What is red/obstacle
    red_mask = cv2.inRange(HSVImage, red_low, red_high) 

    # Range of brown/inner obstacle
    #brown_low = np.array([10, 50, 40], np.uint8) 
    #brown_high = np.array([60, 100, 100], np.uint8) 
    brown_low = np.array([10, 30, 20], np.uint8) 
    brown_high = np.array([60, 100, 140], np.uint8) 

    # What is red/obstacle
    brown_mask = cv2.inRange(HSVImage, brown_low, brown_high) 

  
    #Setting up small goal

    # Range of green/small goal
    #green_lower = np.array([25, 52, 72], np.uint8) 
    #green_upper = np.array([102, 255, 255], np.uint8) 
    green_lower = np.array([25, 50, 70], np.uint8) 
    green_upper = np.array([50, 200, 200], np.uint8) 

    # What is green/Big goal
    green_mask = cv2.inRange(HSVImage, green_lower, green_upper) 
  

    #Setting up big goal

    # Range of blue/big goal
    #blue_lower = np.array([94, 80, 2], np.uint8) 
    #blue_upper = np.array([120, 255, 255], np.uint8) 
    blue_lower = np.array([100, 100, 100], np.uint8) 
    blue_upper = np.array([130, 200, 200], np.uint8) 

    # What is blue/Big goal
    blue_mask = cv2.inRange(HSVImage, blue_lower, blue_upper) 


## Remove noice
    cleanImage = np.ones((5, 5), "uint8") 


## Detect colour from image
    
    masks = []
    colour = []
    display = []
    
    # Detect white/Ball1
    white_mask = cv2.dilate(white_mask, cleanImage) 
    masks.append(white_mask)
    colour.append([0, 0, 0])
    display.append("White/Ball1")
    white_detected = cv2.bitwise_and(image, image,  
                              mask = white_mask) 
    
    # Detect orange/Ball2
    orange_mask = cv2.dilate(orange_mask, cleanImage) 
    masks.append(orange_mask)
    colour.append([255, 255, 0])
    display.append("Orange/Ball2")
    orange_detected = cv2.bitwise_and(image, image,  
                              mask = orange_mask) 
    
    # Detect red/Obstacle
    red_mask = cv2.dilate(red_mask, cleanImage) 
    masks.append(red_mask)
    colour.append([0, 0, 255])
    display.append("Red/Obstacle")
    red_detected = cv2.bitwise_and(image, image,  
                              mask = red_mask) 
    
       # Detect brown/Inner obstacle
    red_mask = cv2.dilate(brown_mask, cleanImage) 
    masks.append(brown_mask)
    colour.append([6, 100, 100])
    display.append("brown/Obstacle")
    red_detected = cv2.bitwise_and(image, image,  
                              mask = brown_mask) 
    
    # Detect grenn/small goal
    green_mask = cv2.dilate(green_mask, cleanImage) 
    masks.append(green_mask)
    colour.append([13, 252, 0])
    display.append("Green/Small goal")
    green_detected = cv2.bitwise_and(image, image,  
                              mask = green_mask) 
    
    # Detect blue/Big goal
    blue_mask = cv2.dilate(blue_mask, cleanImage) 
    masks.append(blue_mask)
    colour.append([0, 101, 252])
    display.append("Blue/Big goal")
    blue_detected = cv2.bitwise_and(image, image,  
                              mask = blue_mask) 
    

## Placing boxes around colours
    for i in range (len(masks)):
        boxColour(image, masks[i], colour[i], display[i])


## End program
    cv2.imshow("Multiple Color Detection in Real-TIme", image) 
    if cv2.waitKey(10) & 0xFF == ord('q'): 
        #cap.release() 
        cv2.destroyAllWindows() 
        break

