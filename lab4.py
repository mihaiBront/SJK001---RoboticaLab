import GUI
import HAL as Drone
import utm

import numpy as np
import cv2
import random

DEBUG = True


def geoToCartesian(geo):
    if DEBUG:
        print("geoToCartesian")

    _cart = geo.copy()
    _cart[:, 1] *= 1 / 60  # divide minutes by 60
    _cart[:, 2] *= 1 / 3600  # divide seconds by 60²

    _cart = np.sum(_cart, axis=1)  # apply a sum in x->

    # convert to UTM and take X and Y only
    return np.array(list(utm.from_latlon(_cart[0], _cart[1]))[0:2])


# ------------------------------------------
#    VARIABLES AND INITIAL DECLARATIONS
# ------------------------------------------

haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# variables for navigation
DRONE_HEIGHT = 5

# known safety boat location, stored as DEG,MIN,SEC
BOAT_GEO_LAT_LONG = np.array([[40, 16, 48.2], [-3, -49, -03.5]])
SURVIVORS_GEO_LAT_LONG = np.array([[40, 16, 47.23], [-3, -49, -01.78]])


BOAT_GEO_CART = geoToCartesian(BOAT_GEO_LAT_LONG)
SURVIVORS_GEO_CART = geoToCartesian(SURVIVORS_GEO_LAT_LONG)

SURVIVORS_FROM_DRONE = SURVIVORS_GEO_CART - BOAT_GEO_CART

current_pos = Drone.get_position()

print(f"current_pos = {current_pos}, survivors from here = {SURVIVORS_FROM_DRONE}")

# variables for free_roam
FREEROAM_SET_POINT_POSITION = SURVIVORS_FROM_DRONE

# ------------------
#    FUNCTIONS
# ------------------


def rotate_image(image, angle):
    if DEBUG:
        print("rotate_image")

    if angle == 0:
        return image

    height, width = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 0.9)
    result = cv2.warpAffine(image, rot_mat, (width, height), flags=cv2.INTER_LINEAR)
    return result


def detect_faces(image_gray):
    if DEBUG:
        print("detect_faces")

    faces_rect = []

    for angle in np.linspace(0, 359, 5):
        image_rot = rotate_image(image_gray, angle)

        faces_rect = haar_cascade.detectMultiScale(image_rot, 1.1, 9)

        if len(faces_rect) > 0:
            for x, y, w, h in faces_rect:
                image_rot = cv2.rectangle(
                    image_rot, (x, y), (x + w, y + h), (0, 255, 0), thickness=2
                )
                image_gray = rotate_image(image_rot, -angle)
            break

    return faces_rect


def random_point_in_circle(radius, centroid):
    if DEBUG:
        print("random_point_in_circle")

    # random angle
    alpha = 2 * np.pi * random.random()

    # random radius
    r = radius * np.sqrt(random.random())

    # calculating coordinates
    x = r * np.cos(alpha) + centroid[0]
    y = r * np.sin(alpha) + centroid[1]

    return [x, y]


def navigation(current_position, setpoint_position, freeroam_radius, freeroam_centroid):

    # calculate distance to survivors position
    distance_to_position = setpoint_position - current_position
    distance_to_position = np.sqrt(np.sum(distance_to_position**2))

    if distance_to_position > freeroam_radius:
        if DEBUG:
            print("target: survivors")

        # navigate towards survivors position
        Drone.set_cmd_pos(setpoint_position[0], setpoint_position[1], DRONE_HEIGHT, 0.6)
    else:
        if DEBUG:
            print("target: freeroam")
        # freeroam in a circle around the survivors position

        # calculate distance from drone to position
        distance_to_freeroam_center = freeroam_centroid - current_position
        distance_to_freeroam_center = np.sqrt(np.sum(distance_to_position**2))

        # if close to randomly selected point
        if distance_to_freeroam_center < 0.75:  # meters
            freeroam_centroid = random_point_in_circle(
                freeroam_radius, setpoint_position
            )
            print(f"NEW FREEROAM DESTINATION {freeroam_centroid}")

        Drone.set_cmd_pos(
            freeroam_centroid[0],
            freeroam_centroid[1],
            DRONE_HEIGHT,
            0.6,
        )


# -------------------
#    SETUP CODE
# -------------------

Drone.takeoff(5)

while True:
    drone_position = Drone.get_position()

    navigation(drone_position[:2], SURVIVORS_FROM_DRONE, 5, FREEROAM_SET_POINT_POSITION)

    ventralImg = cv2.cvtColor(Drone.get_ventral_image(), cv2.COLOR_BGR2GRAY)
    """faces_list = detect_faces(ventralImg)

    if len(faces_list) > 0:
        print("detected faces")"""

    GUI.showImage(Drone.get_frontal_image())
    GUI.showLeftImage(ventralImg)
