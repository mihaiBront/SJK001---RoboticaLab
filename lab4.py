import GUI
import HAL as Drone
import utm

import numpy as np

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

while True:
    GUI.showImage(Drone.get_frontal_image())
    GUI.showLeftImage(Drone.get_ventral_image())

    Drone.set_cmd_pos(SURVIVORS_FROM_DRONE[0], SURVIVORS_FROM_DRONE[1], 3, 0.6)
    print(current_pos)
