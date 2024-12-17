import GUI
import HAL as Drone
import utm

import numpy as np
import cv2
import random

DEBUG = False


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

# number of victims for knowing when to go back to base
VICTIMS_COUNT = 6

# variables for navigation
DRONE_HEIGHT = 2

# known safety boat location, stored as DEG,MIN,SEC
BOAT_GEO_LAT_LONG = np.array([[40, 16, 48.2], [-3, -49, -03.5]])
SURVIVORS_GEO_LAT_LONG = np.array([[40, 16, 47.23], [-3, -49, -01.78]])


BOAT_GEO_CART = geoToCartesian(BOAT_GEO_LAT_LONG)
SURVIVORS_GEO_CART = geoToCartesian(SURVIVORS_GEO_LAT_LONG)

SURVIVORS_FROM_DRONE = SURVIVORS_GEO_CART - BOAT_GEO_CART

if DEBUG:
    SURVIVORS_FROM_DRONE = np.array([5.0, 5.0])

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

    for angle in np.linspace(0, 360, 5):
        image_rot = rotate_image(image_gray, angle)

        faces_rect = haar_cascade.detectMultiScale(
            image_rot, scaleFactor=1.1, minNeighbors=4
        )

        if len(faces_rect) > 0:
            print("\t det face...")
            return faces_rect

    return []


def store_faces(
    list_faces_position: list,
    new_face_position: list[float],
    minimum_face_separation_m: float,
):
    if len(list_faces_position) == 0:
        return [new_face_position]

    new_list = list_faces_position.copy()

    for face_position in list_faces_position:
        distance_known_face = new_face_position - face_position
        distance_known_face = np.sqrt(np.sum(distance_known_face**2))

        if distance_known_face < minimum_face_separation_m:
            return new_list

    new_list.append(new_face_position)
    return new_list


def random_point_in_circle(radius, centroid, current_position):
    if DEBUG:
        pass

    distance_to_random_point = 0
    point = np.array([0, 0])

    while distance_to_random_point < radius / 2:

        # calculating coordinates
        point = np.array(
            [
                np.random.uniform(-radius, radius) + centroid[0],
                np.random.uniform(-radius, radius) + centroid[1],
            ]
        )

        distance_to_random_point = current_position - point
        distance_to_random_point = np.sqrt(np.sum(distance_to_random_point**2))
        print(f"calc_dist = {distance_to_random_point} ({point})")

    return point


def navigation(
    current_position,
    setpoint_position,
    freeroam_radius,
    freeroam_centroid,
    list_detected_faces,
):

    # calculate distance to survivors position
    distance_to_position = setpoint_position - current_position
    distance_to_position = np.sqrt(np.sum(distance_to_position**2))

    if len(list_detected_faces) > VICTIMS_COUNT:
        Drone.set_cmd_pos(
            0.0,
            0.0,
            DRONE_HEIGHT,
            0.6,
        )
        return freeroam_centroid

    if distance_to_position > freeroam_radius * 1.2:
        if DEBUG:
            print("target: survivors")

        # navigate towards survivors position
        Drone.set_cmd_pos(setpoint_position[0], setpoint_position[1], DRONE_HEIGHT, 0.6)
        freeroam_centroid = setpoint_position

    else:
        # freeroam in a circle around the survivors position

        # calculate distance from drone to position
        distance_to_freeroam_center = freeroam_centroid - current_position
        distance_to_freeroam_center = np.sqrt(np.sum(distance_to_freeroam_center**2))

        if DEBUG:
            pass
        # if close to randomly selected point
        if distance_to_freeroam_center < 0.09:  # meters
            print(
                f"OLD FREEROAM DESTINATION {freeroam_centroid} {distance_to_freeroam_center}m away"
            )
            freeroam_centroid = random_point_in_circle(
                freeroam_radius, setpoint_position, current_position
            )
            print(f"NEW FREEROAM DESTINATION {freeroam_centroid}")

        Drone.set_cmd_pos(
            freeroam_centroid[0],
            freeroam_centroid[1],
            DRONE_HEIGHT,
            0.6,
        )

    return freeroam_centroid


# -------------------
#    SETUP CODE
# -------------------

Drone.takeoff(DRONE_HEIGHT)

list_faces = []

while True:
    drone_position = np.array(Drone.get_position())

    FREEROAM_SET_POINT_POSITION = navigation(
        drone_position[:2],
        SURVIVORS_FROM_DRONE,
        5,
        FREEROAM_SET_POINT_POSITION,
        list_faces,
    )

    ventralImg = cv2.cvtColor(Drone.get_ventral_image(), cv2.COLOR_BGR2GRAY)

    distance_to_survivors = drone_position[:2] - SURVIVORS_FROM_DRONE[:2]
    distance_to_survivors = np.sqrt(np.sum(distance_to_survivors**2))

    if distance_to_survivors < 12:
        faces_list = detect_faces(ventralImg)

        if len(faces_list) > 0:
            list_faces = store_faces(list_faces, drone_position[:2], 3.0)
            print(f"detected {len(faces_list)} faces; current list is {list_faces}")

    GUI.showImage(Drone.get_frontal_image())
    GUI.showLeftImage(ventralImg)
