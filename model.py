import random
import numpy as np

from mesa import Model
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from car import Car
from qcar import QCar
from deepqcar import DeepQCar
from barrier import Barrier

import util
from settings import AgentType


def get_rewards_sum(model):
    return model.agent.rewards_sum

class ChaosModel(Model):
    '''
    The stochastic highway simulation model
    '''

    def __init__(self, agent_type, width=500, height=500, 
                 num_adversaries=8, road_width=60):
        self.num_adversaries = num_adversaries
        self.road_width = road_width
        self.schedule = RandomActivation(self)

        self.space = ContinuousSpace(width, height, True)
        self.cars = []

        self.make_agents(AgentType(agent_type))
        self.running = True

        self.datacollector = DataCollector(
            model_reporters={"Agent rewards sum": get_rewards_sum})

    @property
    def learn_agent(self):
        return self.cars[0]

    def is_overlapping(self, x, y, width, length):
        for other in self.cars:
            if x - width / 2 <= other.pos[0] + other.width / 2 and \
                    x + width / 2 >= other.pos[0] - other.width / 2 and \
                    y + length / 2 >= other.pos[1] - other.length / 2 and \
                    y - length / 2 <= other.pos[1] + other.length / 2:
                return True
        return False

    def adversary(self, unique_id):
        # Initial speed and heading
        speed = util.rand_min_max(0, 3)
        heading = np.radians(-90)

        # Randomly add large (blue), medium (orange), or small (green) car
        val = random.random()
        if val < 0.33:
            color = "Blue"
            target_speed = util.rand_min_max(2, 4)
            car_width = util.rand_min_max(8, 9)
            car_length = util.rand_min_max(35, 45)
        elif val < 0.66:
            color = "Orange"
            target_speed = util.rand_min_max(3, 6)
            car_width = util.rand_min_max(7, 8)
            car_length = util.rand_min_max(16, 30)
        else:
            color = "Green"
            target_speed = util.rand_min_max(6, 7)
            car_width = util.rand_min_max(5, 6)
            car_length = util.rand_min_max(12, 16)

        x = util.rand_center_spread(self.space.x_max/2, self.road_width)
        y = random.uniform(self.space.y_min + 1, self.space.y_max - 1)
        while self.is_overlapping(x, y, car_width, car_length):
            x = util.rand_center_spread(self.space.x_max/2, self.road_width)
            y = random.uniform(self.space.y_min + 1, self.space.y_max - 1)
        pos = np.array((x, y))

        return Car(unique_id, self, pos, speed, heading, color, target_speed,
                   width=car_width, length=car_length)

    def make_agents(self, agent_type):
        '''
        Add all agents to model space
        '''

        # Learning agent
        if agent_type == AgentType.DEEPQ:
            learn_agent = DeepQCar
        elif agent_type == AgentType.QLEARN:
            learn_agent = QCar
        else: # agent_type == AgentType.BASIC
            learn_agent = Car
        pos = np.array((self.space.x_max/2, self.space.y_max-1))
        speed = 0
        heading = np.radians(-90)
        target_speed = 10
        color = "Black"
        car_width = 6
        car_length = 12
        qcar = learn_agent(0, self, pos, speed, heading, 
            color, target_speed, length=car_length, width=car_width)
        self.agent = qcar
        self.cars.append(qcar)
        self.space.place_agent(qcar, pos)
        self.schedule.add(qcar)

        # Adversaries
        self.add_frozen(1)
        for i in range(2, self.num_adversaries + 1):
            car = self.adversary(i)
            self.cars.append(car)
            self.space.place_agent(car, car.pos)
            self.schedule.add(car)

        # Barriers
        self.add_barrier(len(self.cars), (self.space.x_max + self.road_width + 10) / 2)
        self.add_barrier(len(self.cars) + 1, (self.space.x_max - self.road_width - 10) / 2)
        
    def add_frozen(self, unique_id):
        pos = np.array((250, 250))
        car = Car(unique_id, self, pos, 0, np.radians(-90), "Indigo", 0, width=6, length=12)
        self.cars.append(car)
        self.space.place_agent(car, pos)
        self.schedule.add(car)

    def add_barrier(self, unique_id, x):
        y = self.space.y_max / 2
        barrier = Barrier(unique_id, self, np.array([x, y]), "Black", 1, self.space.y_max)
        self.schedule.add(barrier)

    def step(self):
        self.datacollector.collect(self)
        # First loop through and have all cars choose an action
        # before the action is actually propagated forward
        for car in self.cars:
            car.choose_action()

        # Propagate forward one step based on chosen actions
        self.schedule.step()    
