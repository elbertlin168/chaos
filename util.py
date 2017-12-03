import random
import numpy as np

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
