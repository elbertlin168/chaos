import random
import numpy as np

from mesa import Model
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation

from car import Car

import util

class ChaosModel(Model):
    '''
    '''

    def __init__(self, canvas_size=500, num_adversaries=50, road_width=40):
        '''
        '''
        self.num_adversaries = num_adversaries
        self.road_width = road_width
        self.schedule = RandomActivation(self)

        self.space = ContinuousSpace(canvas_size, canvas_size, True)
        self.cars = []

        self.make_agents()
        self.running = True


    def reward(self, current_speed, risk, collided):
        speed_reward = (current_speed < max_speed * 1.1) * \
                        (1.0 * current_speed / self.max_speed) * 500
        speed_cost = - (current_speed > max_speed * 1.1) * \
                        (1.0 * current_speed / self.max_speed) * 800
        risk_cost = -risk * 200
        collision_cost = collided * -50000
        return speed_reward + speed_cost + risk_cost + collision_cost

    def make_agents(self):
        '''
        '''
        for i in range(self.num_adversaries):

            # Random start position
            x = util.rand_center_spread(self.space.x_max/2, self.road_width)

            # Space out start positions in y coordinate so agents don't overlap
            # at initialization
            y = self.space.y_max*i/self.num_adversaries

            pos = np.array((x, y))

            # Initial speed and heading
            speed = 0
            heading = np.radians(-90)

            # Random target speed
            target_speed = util.rand_min_max(3, 8)

            # if i == 0:
            #     pos = np.array((250,250))
            #     speed = 5
            #     target_speed = 5
            #     heading = np.radians(-90)
            # elif i == 1:
            #     pos = np.array((250, 490))
            #     speed = 15
            #     target_speed = 15
            #     heading = np.radians(-90)

            # Initialize car
            car = Car(i, self, pos, speed, heading, self.road_width, target_speed)
            self.cars.append(car)
            self.space.place_agent(car, pos)
            self.schedule.add(car)


    def step(self):
        # First loop through and have all cars choose an action
        # before the action is actually propagated forward
        for car in self.cars:
            car.choose_action()

        # Propagate forward one step based on chosen actions
        self.schedule.step()


