import random
import numpy as np

from mesa import Model
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation

from car import Car


class ChaosModel(Model):
    '''
    '''

    def __init__(self, lanes=5, num_adversaries=10, max_speed=5):
        '''
        '''
        self.lanes = lanes
        self.num_adversaries = num_adversaries
        self.max_speed = max_speed
        self.schedule = RandomActivation(self)
        self.space = ContinuousSpace(lanes * 4, 500, True)
        self.make_agents()
        self.running = True

    def make_agents(self):
        '''
        '''
        for i in range(self.num_adversaries):
            x = random.random() * self.space.x_max
            y = random.random() * self.space.y_max
            pos = np.array((x, y))
            speed = random.random() * self.max_speed
            car = Car(i, self, pos, speed)
            self.space.place_agent(car, pos)
            self.schedule.add(car)

    def step(self):
        self.schedule.step()
