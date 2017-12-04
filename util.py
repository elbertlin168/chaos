import random
import numpy as np

from settings import SAFETY_MARGIN

def rand_min_max(a, b):
    spread = b - a
    return random.random() * spread + a

def rand_center_spread(center, spread):
    a = center - spread / 2
    return random.random() * spread + a

def wrap_angle(angles):
    '''
    Returns angle betwee -pi and pi
    '''
    return (angles + np.pi) % (2 * np.pi) - np.pi

def is_overlapping(x, y, width, length, others, margin=SAFETY_MARGIN):
    for other in others:
        h_margin = (width + other.width) * (1 + margin) / 2
        v_margin = (length + other.length) * (1 + margin) / 2

        if abs(x - other.pos[0]) - h_margin < 0 and \
                abs(y - other.pos[1]) - v_margin < 0:
            return True

    return False

def get_bin(val, v_min, bin_size):
    return int((val - v_min) // bin_size)
