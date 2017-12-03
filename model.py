import random
import numpy as np

from mesa import Model
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from car import Car
from qcar import QCar
from barrier import Barrier

import util

def get_rewards_sum(model):
    return model.agent.rewards_sum

class ChaosModel(Model):
    '''
    '''

    def __init__(self, agent_type, canvas_size=500, 
                 num_adversaries=8, road_width=60):
        '''
        '''
        self.num_adversaries = num_adversaries
        self.road_width = road_width
        self.schedule = RandomActivation(self)

        self.space = ContinuousSpace(canvas_size, canvas_size, True)
        self.cars = []
        self.agent = []

        self.make_agents(canvas_size)
        self.running = True

        self.datacollector = DataCollector(
            model_reporters={"Agent rewards sum": get_rewards_sum})


    def make_agents(self, canvas_size):
        '''
        '''

        # Qcar
        pos = np.array((self.space.x_max/2, self.space.y_max-1))
        speed = 0
        heading = np.radians(-90)
        target_speed = 10
        color = "Black"
        car_width = 6
        car_length = 12
        qcar = QCar(0, self, pos, speed, heading, self.road_width, 
            color, target_speed, car_length=car_length, car_width=car_width)
        self.agent = qcar
        self.cars.append(qcar)
        self.space.place_agent(qcar, pos)
        self.schedule.add(qcar)


        for i in range(1, self.num_adversaries + 1):

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

            if i == 1:
                pos = np.array((250, 250))
                speed = 0
                target_speed = 0
                heading = np.radians(-90)
                car_width = 6
                car_length = 12

            # Initialize car
            car = Car(i, self, pos, speed, heading, self.road_width,
                color, target_speed, car_length=car_length, car_width=car_width)


            self.cars.append(car)
            self.space.place_agent(car, pos)
            self.schedule.add(car)


        # Barriers
        color = "Black"
        width = 1
        length = canvas_size
        y = self.space.y_max / 2
        x = (self.space.x_max + self.road_width + 10) / 2
        pos = np.array((x, y))
        i = i + 1
        barrier = Barrier(i, self, pos, color, width, length)
        self.space.place_agent(barrier, pos)
        self.schedule.add(barrier)

        x = (self.space.x_max - self.road_width - 10) / 2
        pos = np.array((x, y))
        i = i + 1
        barrier = Barrier(i, self, pos, color, width, length)
        self.space.place_agent(barrier, pos)
        self.schedule.add(barrier)


    def step(self):
        self.datacollector.collect(self)
        # First loop through and have all cars choose an action
        # before the action is actually propagated forward
        for car in self.cars:
            car.choose_action()

        # Propagate forward one step based on chosen actions
        self.schedule.step()    
