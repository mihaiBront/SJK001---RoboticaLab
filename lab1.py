import GUI
import HAL

import cv2 as cv
import numpy as np

# i = 0

while True:
    img = HAL.getImage()
    
    # masking red to get only the line:
    img_mask = cv.inRange(cv.cvtColor(img, cv.COLOR_BGR2HSV),
                          (0, 125, 125),(30, 255,255))
                          
    # image segmentation
    c, _ = cv.findContours(img_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    
    # finding centroid of the vision of the line
    M = cv.moments(c[0])
    
    cX, cY = 0, 0
    if M["m00"] != 0:
        cX = M["m10"] / M["m00"]
        cY = M["m01"] / M["m00"]
        
    
    if cX > 0:
        e = img_mask.shape[1]/2 - cX
        HAL.setV(4)
        HAL.setW(0.0056*e)
    else:
        HAL.setV(0)
    
    cv.drawMarker(img_mask, (int(cX), int(cY)), 0, thickness=2)
    GUI.showImage(img_mask)

