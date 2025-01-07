import GUI
import HAL as Drone
import utm

import numpy as np
import cv2
import random

DEBUG = False  # variable for activating extra prints for debug

"""
This problem can be solved more simplistically by implementing a simple state machine defined
by the following grafcet (starting from the moment after the takeoff is performed):

       +-----+
       |+---+|  ACTION:
       || 0 ||  Initial state: Go towards the victims position
       |+---+|
       +-----+
          |     TRANSITION:
        --+---> Distance to the victims < Threshold (arbitrary one; when close to victims)
          |
        +---+   ACTION
        | 1 |   Freeroam around the reported victims position
        +---+
          |     TRANSITION:
        --+---> Found Victims == Expected Victims (6, in this case)
          |
        +---+   ACTION:
        | 2 |   Return to the boat (I assume there will be someone there
        +---+   waiting for the drone and will catch it mid flight, no landing
                required)
"""


# region MISC METHODS
def geoToCartesian(geo):
    """Method for converting from the boat geodesic coordinates to the dron-relative position

    Args:
        geo (list[float]): Geodesic coordinates

    Returns:
        list[float]: Drone coordinates
    """
    if DEBUG:
        print("geoToCartesian")

    _cart = geo.copy()
    _cart[:, 1] *= 1 / 60  # divide minutes by 60
    _cart[:, 2] *= 1 / 3600  # divide seconds by 60²

    _cart = np.sum(_cart, axis=1)  # apply a sum in x->

    # convert to UTM and take X and Y only
    return np.array(list(utm.from_latlon(_cart[0], _cart[1]))[0:2])


def spiral_patrol(center, num_loops, radius_increment, num_points_per_loop=100):
    """
    Generates a spiral patrol path around a point.

    Parameters:
        center (tuple): Target position (x, y).
        num_loops (int): Number of loops in the spiral.
        radius_increment (float): Increase in radius per loop.
        num_points_per_loop (int): Number of points per loop.

    Returns:
        list: List of (x, y) points representing the patrol path.
    """
    x_c, y_c = center
    points = []
    theta = 0
    for _ in range(num_loops):
        for i in range(num_points_per_loop):
            r = radius_increment * theta / (2 * np.pi)
            x = x_c + r * np.cos(theta)
            y = y_c + r * np.sin(theta)
            points.append((x, y))
            theta += 2 * np.pi / num_points_per_loop
    return points


# endregion

# region BASELINE VARIABLES AND OBJECTS INITIALIZATION

haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# number of victims for knowing when to go back to base
VICTIMS_COUNT = 6

# variables for navigation
DRONE_HEIGHT = 2

# known safety boat location, stored as DEG,MIN,SEC
_boat_geo_latLong = np.array([[40, 16, 48.2], [-3, -49, -03.5]])
_survivors_geo_latLong = np.array([[40, 16, 47.23], [-3, -49, -01.78]])

_boat_geo_cartesian = geoToCartesian(_boat_geo_latLong)
_survivors_geo_cartesian = geoToCartesian(_survivors_geo_latLong)

# variables for navigation
SURVIVORS_FROM_DRONE = (_survivors_geo_cartesian - _boat_geo_cartesian)[:2]

if DEBUG:
    SURVIVORS_FROM_DRONE = np.array([5, 5])

FREEROAM_SET_POINT_POSITION = SURVIVORS_FROM_DRONE
BOAT_POSITION = np.array([0, 0])

# patrolling pattern for the drone when freeroaming around the victims position
PATROL_PATTERN = spiral_patrol(
    SURVIVORS_FROM_DRONE, num_loops=10, radius_increment=2, num_points_per_loop=6
)
PATROL_PATTERN_INDEX = 0

# states machine definition
#               TO-VICTIMS  FREEROAM  BACK TO BOAT
STATEMACHINE = [True, False, False]  # initialize it as
TO_VICTIMS = 0
FREEROAM = 1
BACK_TO_BOAT = 2

# takeoff drone
Drone.takeoff(DRONE_HEIGHT)

# initialize a list to store faces
list_faces = []

# endregion


def _rotate_image(image, angle):
    if angle == 0:
        return image

    height, width = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    result = cv2.warpAffine(image, rot_mat, (width, height), flags=cv2.INTER_LINEAR)
    return result


def detect_faces(image_gray):
    for angle in range(0, 360, 10):
        image_rot = _rotate_image(image_gray, angle)

        faces_rect = haar_cascade.detectMultiScale(
            image_rot, scaleFactor=1.1, minNeighbors=4
        )

        if len(faces_rect) > 0:
            return True

    return False


def updateFaces(image_gray, drone_position, lst):
    new_lst = lst.copy()

    if not detect_faces(image_gray):
        return new_lst

    if len(new_lst) > 0:
        for face_pos in new_lst:
            if np.linalg.norm(drone_position - face_pos) < 3.5:
                return new_lst

    new_lst.append(np.array(drone_position))
    print(f"New face detected {new_lst}")
    return new_lst


def freeroamTarget(drone_position, patrol_pattern, patrol_pattern_index):
    if np.linalg.norm(drone_position - patrol_pattern[patrol_pattern_index]) < 0.4:
        patrol_pattern_index += 1
        print(f"\t Freeroaming to next path step ({patrol_pattern_index})")
        if patrol_pattern_index >= len(patrol_pattern) - 1:
            patrol_pattern_index = 0
    return patrol_pattern_index


# ----------------------------- MAIN LOOP -----------------------------------
print(f"1: TRAVEL TO REPORTED VICTIMS POSITION ({SURVIVORS_FROM_DRONE})")
while True:
    # 1. sensoric update
    drone_position = np.array(Drone.get_position())[:2]
    ventralImg = cv2.cvtColor(Drone.get_ventral_image(), cv2.COLOR_BGR2GRAY)

    # 2. Transitions calculation
    if (
        STATEMACHINE[TO_VICTIMS]
        and np.linalg.norm(drone_position - SURVIVORS_FROM_DRONE) < 1
    ):
        print("2: FREEROAMING AROUND VICTIMS")
        # if drone is going towards victims and it is close enough to them, go to freeroam
        STATEMACHINE = [False, True, False]
    elif STATEMACHINE[FREEROAM] and len(list_faces) >= VICTIMS_COUNT:
        print("3. RETURNING TO BOAT")
        # if drone is in freeroam and it has found enough victims, go back to boat
        STATEMACHINE = [False, False, True]

    # 3. Actions calculation
    if STATEMACHINE[TO_VICTIMS]:
        Drone.set_cmd_pos(
            SURVIVORS_FROM_DRONE[0],
            SURVIVORS_FROM_DRONE[1],
            DRONE_HEIGHT,
            0.6,
        )
    elif STATEMACHINE[FREEROAM]:
        # try and find faces
        list_faces = updateFaces(ventralImg, drone_position, list_faces)

        # update freeroam position
        PATROL_PATTERN_INDEX = freeroamTarget(
            drone_position=drone_position,
            patrol_pattern=PATROL_PATTERN,
            patrol_pattern_index=PATROL_PATTERN_INDEX,
        )
        FREEROAM_SET_POINT_POSITION = PATROL_PATTERN[PATROL_PATTERN_INDEX]

        Drone.set_cmd_pos(
            FREEROAM_SET_POINT_POSITION[0],
            FREEROAM_SET_POINT_POSITION[1],
            DRONE_HEIGHT,
            0.6,
        )
    elif STATEMACHINE[BACK_TO_BOAT]:
        Drone.set_cmd_pos(BOAT_POSITION[0], BOAT_POSITION[1], DRONE_HEIGHT, 0.6)

    # 4. Other outputs
    GUI.showImage(Drone.get_frontal_image())
    GUI.showLeftImage(ventralImg)
