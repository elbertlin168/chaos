import random

def rand_min_max(a, b):
    spread = b - a
    return random.random()*spread + a

def rand_center_spread(center, spread):
    a = center - spread/2
    return random.random()*spread + a

