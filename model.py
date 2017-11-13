import random
import numpy as np

from mesa import Model
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation

from car import Car


class ChaosModel(Model):
    '''
    '''

    def __init__(self, lanes=5, num_adversaries=50, max_speed=30):
        '''
        '''
        self.lanes = lanes
        self.num_adversaries = num_adversaries
        self.max_speed = max_speed
        self.schedule = RandomActivation(self)

        # self.space = ContinuousSpace(lanes * 4, 500, True)
        # Just make space square since the display is square
        self.space = ContinuousSpace(500, 500, True)
        
        self.make_agents()
        self.running = True

    def make_agents(self):
        '''
        '''
        for i in range(self.num_adversaries):

            # Random start position
            x = random.random() * self.space.x_max
            y = random.random() * self.space.y_max
            pos = np.array((x, y))

            # Random speed 
            speed = random.random() * self.max_speed

            # Random heading
            heading = np.radians(random.random()*360 - 180)

            # Initialize car
            car = Car(i, self, pos, speed, heading)
            self.space.place_agent(car, pos)
            self.schedule.add(car)

    def step(self):
        self.schedule.step()
