import GUI
import HAL as Drone
import utm

import numpy as np
import cv2

haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# known safety boat location, stored as DEG,MIN,SEC
BOAT_GEO_LAT_LONG = np.array([[40, 16, 48.2], [-3, -49, -03.5]])
SURVIVORS_GEO_LAT_LONG = np.array([[40, 16, 47.23], [-3, -49, -01.78]])


def geoToCartesian(geo):
    _cart = geo.copy()
    _cart[:, 1] *= 1 / 60  # divide minutes by 60
    _cart[:, 2] *= 1 / 3600  # divide seconds by 60²

    _cart = np.sum(_cart, axis=1)  # apply a sum in x->

    # convert to UTM and take X and Y only
    return np.array(list(utm.from_latlon(_cart[0], _cart[1]))[0:2])


BOAT_GEO_CART = geoToCartesian(BOAT_GEO_LAT_LONG)
SURVIVORS_GEO_CART = geoToCartesian(SURVIVORS_GEO_LAT_LONG)

SURVIVORS_FROM_DRONE = SURVIVORS_GEO_CART - BOAT_GEO_CART

current_pos = Drone.get_position()

print(f"current_pos = {current_pos}, survivors from here = {SURVIVORS_FROM_DRONE}")

Drone.takeoff(3)
desiredPosition = SURVIVORS_FROM_DRONE

while True:
    drone_position = Drone.get_position()
    distance_to_position = SURVIVORS_FROM_DRONE[:2] - drone_position[:2]
    distance_to_position = np.sqrt(np.sum(distance_to_position**2))

    if distance_to_position < 2:
        print("drone should stop")

    ventralImg = Drone.get_ventral_image()
    ventralImg_gray = cv2.cvtColor(ventralImg, cv2.COLOR_BGR2GRAY)

    faces_rect = haar_cascade.detectMultiScale(ventralImg_gray, 1.1, 9)

    if len(faces_rect) > 0:
        print("----------------FACES DETECTED----------------")
        for x, y, w, h in faces_rect:
            print("Face detected! (pos: {drone_position}); imCoords: {(x,y)}")
            cv2.rectangle(ventralImg, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

    GUI.showImage(Drone.get_frontal_image())
    GUI.showLeftImage(ventralImg)
    Drone.set_cmd_pos(desiredPosition[0], desiredPosition[1], 5, 0.9)
