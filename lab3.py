import GUI
import HAL

import cv2 as cv
import time

kP = 6.7e-3
kD = 6.5e-4
kI = 6.5e-4

tLast = time.time()
tCurr = time.time()

_i = 0  # initialize integrative term
windup_limit = 1000  # high so it does not apply right now

eLast = 0.0

spSpeed = 4

while True:
    ## CALCULATE ERROR:
    img = HAL.getImage()[230:-110, :, :] 
    # setting a roi on the captured image so the centroid is 
    # calculated with more anticipation, potentially some 
    # corner cutting and the cycle time is lower
    img_mask = cv.inRange(
        cv.cvtColor(img, cv.COLOR_BGR2HSV), (0, 125, 125), (30, 255, 255)
    )
    c, _ = cv.findContours(img_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cX, cY = 0, 0

    if len(c) > 0:
        M = cv.moments(c[0])

        if M["m00"] != 0:
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]

    # If centroid is detected, make control, else, jump to next cycle and leave
    # everything as is
    if cX > 0:
        e = img_mask.shape[1] / 2 - cX

        ## CALCULATE DELTA TIME FOR PROCESSING
        tCurr = time.time()
        Dt = tCurr - tLast

        ## CALCULATE PROPORTIONAL TERM:
        _p = kP * e

        ## CALCULATE DERIVATIVE TERM:
        if Dt > 0.0000:
            _d = kD * (e - eLast) / Dt
        else:
            _d = 0.0  # avoid dividing by 0

        ## CALCULATE INTEGRAL TERM:
        _i += kI * e * Dt

        # anti-windup
        # if _i > windup_limit:
        #     _i = windup_limit
        # if _i < -windup_limit:
        #     _i = -windup_limit

        ## UPDATE CYCLE VARIABLES:
        eLast = e
        tLast = tCurr

        ## CALCULATE NEW TURN
        setW = _p + _d + _i
        #print(f"Setw = {_p} + {_d} + {_i} = {setW}")

        HAL.setV(spSpeed)
        HAL.setW(setW)

    cv.drawMarker(img_mask, (int(cX), int(cY)), 0, thickness=2)
    GUI.showImage(img_mask)
