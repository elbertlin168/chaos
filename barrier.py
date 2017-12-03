import numpy as np
from mesa import Agent

class Barrier(Agent):
    '''
    Basic barrier agent for drawing
    '''

    def __init__(self, unique_id, model, 
                 pos, color, width, length):
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.color = color
        self.width = width
        self.length = length
