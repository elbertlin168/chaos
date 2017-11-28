import numpy as np
import random
from mesa import Agent
import math
import util

class Barrier(Agent):
    '''
    '''

    def __init__(self, 
                 unique_id, 
                 model, 
                 pos, 
                 color,
                 car_width, 
                 car_length
                 ):
        '''
        '''
        super().__init__(unique_id, model)

        # Initial position
        self.pos = np.array(pos)

        # Set orig color
        self.orig_color = color
        self.color = color

        # Size of car for collision detection
        self.car_width = car_width
        self.car_length = car_length





