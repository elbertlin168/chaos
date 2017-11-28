import random
import numpy as np

from mesa import Model
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation

from car import Car
from barrier import Barrier

import util

class ChaosModel(Model):
    '''
    '''

    def __init__(self, canvas_size=500, num_adversaries=8, road_width=60):
        '''
        '''
        self.num_adversaries = num_adversaries
        self.road_width = road_width
        self.schedule = RandomActivation(self)

        self.space = ContinuousSpace(canvas_size, canvas_size, True)
        self.cars = []

        self.make_agents(canvas_size)
        self.running = True


    def reward(self, current_speed, risk, collided):
        speed_reward = (current_speed < max_speed * 1.1) * \
                        (1.0 * current_speed / self.max_speed) * 500
        speed_cost = - (current_speed > max_speed * 1.1) * \
                        (1.0 * current_speed / self.max_speed) * 800
        risk_cost = -risk * 200
        collision_cost = collided * -50000
        return speed_reward + speed_cost + risk_cost + collision_cost

    def make_agents(self, canvas_size):
        '''
        '''
        for i in range(self.num_adversaries + 1):

            # Random start position
            x = util.rand_center_spread(self.space.x_max/2, self.road_width)

            # Space out start positions in y coordinate so agents don't overlap
            # at initialization
            y = self.space.y_max*i/(self.num_adversaries + 1)

            pos = np.array((x, y))

            # Initial speed and heading
            speed = 0
            heading = np.radians(-90)

            # Random target speed
            val = random.random()
            if val < 0.33:
                target_speed = util.rand_min_max(2, 4)
                color = "Blue"
                car_width = util.rand_min_max(8, 9)
                car_length = util.rand_min_max(35, 45)
            elif val < 0.66:
                target_speed = util.rand_min_max(3, 6)
                color = "Orange"
                car_width = util.rand_min_max(7, 8)
                car_length = util.rand_min_max(16, 30)
            else:
                target_speed = util.rand_min_max(6, 7)
                color = "Green"
                car_width = util.rand_min_max(5, 6)
                car_length = util.rand_min_max(12, 16)

            if i == 0:
                target_speed = 10
                color = "Black"
                car_width = 6
                car_length = 12

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
            car = Car(i, self, pos, speed, heading, self.road_width, 
                color, target_speed, car_length=car_length, car_width=car_width)


            self.cars.append(car)
            self.space.place_agent(car, pos)
            self.schedule.add(car)



        # Barrier
        color = "Black"
        car_width = 1
        car_length = canvas_size
        y = self.space.y_max/2


        x = self.space.x_max/2 + self.road_width
        pos = np.array((x, y))

        barrier = Barrier(i, self, pos, 
            color, car_length=car_length, car_width=car_width)

        self.space.place_agent(barrier, pos)
        self.schedule.add(barrier)

        x = self.space.x_max/2 - self.road_width
        pos = np.array((x, y))

        barrier = Barrier(i, self, pos, 
            color, car_length=car_length, car_width=car_width)

        self.space.place_agent(barrier, pos)
        self.schedule.add(barrier)


    def step(self):
        # First loop through and have all cars choose an action
        # before the action is actually propagated forward
        for car in self.cars:
            car.choose_action()

        # Propagate forward one step based on chosen actions
        self.schedule.step()


