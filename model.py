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
    return model.learn_agent.rewards_sum

class ChaosModel(Model):
    '''
    The stochastic highway simulation model
    '''

    #################################################################
    ## INITIALIZATION FUNCTIONS
    #################################################################

    def __init__(self, agent_type, width=500, height=500, 
                 num_adversaries=8, road_width=60, frozen=True):
        self.num_adversaries = num_adversaries
        self.road_width = road_width
        self.schedule = RandomActivation(self)

        self.space = ContinuousSpace(width, height, True)
        self.cars = []

        self.make_agents(AgentType(agent_type), frozen)
        self.running = True

        self.datacollector = DataCollector(
            model_reporters={"Agent rewards sum": get_rewards_sum})

    def make_adversary(self, unique_id):
        # Initial speed and heading
        speed = util.rand_min_max(0, 3)
        heading = np.radians(-90)

        # Randomly add large (blue), medium (orange), or small (green) car
        val = random.random()
        if val < 0.33:
            color = "Blue"
            target_speed = util.rand_min_max(2, 4)
            width = util.rand_min_max(8, 9)
            length = util.rand_min_max(35, 45)
        elif val < 0.66:
            color = "Orange"
            target_speed = util.rand_min_max(3, 6)
            width = util.rand_min_max(7, 8)
            length = util.rand_min_max(16, 30)
        else:
            color = "Green"
            target_speed = util.rand_min_max(6, 7)
            width = util.rand_min_max(5, 6)
            length = util.rand_min_max(12, 16)

        # Add car to a random position if not overlapping within a margin
        # Try (num_adversaries * 2) times before giving up
        pos = None
        for _ in range(self.num_adversaries * 2):
            x = util.rand_center_spread(self.space.x_max/2, self.road_width)
            y = random.uniform(self.space.y_min + 1, self.space.y_max - 1)
            if not util.is_overlapping(x, y, width, length, self.cars):
                pos = np.array((x, y))
                break
        if pos is None:
            return None

        return Car(unique_id, self, pos, speed, heading, color,
                   target_speed=target_speed, width=width, length=length)

    def make_learn_agent(self, agent_type, unique_id):
        if agent_type == AgentType.DEEPQ:
            car = DeepQCar
        elif agent_type == AgentType.QLEARN:
            car = QCar
        else: # agent_type == AgentType.BASIC
            car = Car
        pos = np.array((self.space.x_max/2, self.space.y_max-1))
        return car(unique_id, self, pos, 0, np.radians(-90), "Black",
                  target_speed=10, width=6, length=12)

    def make_frozen(self, unique_id):
        pos = np.array((250, 250))
        return Car(unique_id, self, pos, 0, np.radians(-90), "Indigo",
                  target_speed=0, width=6, length=12)

    def make_agents(self, agent_type, frozen):
        '''
        Add all agents to model space
        '''
        # Car agents
        for i in range(0, self.num_adversaries + 1):
            if i == 0:
                car = self.make_learn_agent(agent_type, i)
            elif frozen:
                car = self.make_frozen(i)
                frozen = False
            else:
                car = self.make_adversary(i)
            if car is None:
                print("WARNING: Could only add %d adversaries" % (i-1))
                break
            self.cars.append(car)
            self.space.place_agent(car, car.pos)
            self.schedule.add(car)

        # Barriers
        x = (self.space.x_max + self.road_width + 10) / 2
        self.add_barrier(len(self.cars), x)
        self.add_barrier(len(self.cars) + 1, x - self.road_width - 10)

    def add_barrier(self, unique_id, x):
        y = self.space.y_max / 2
        barrier = Barrier(unique_id, self, np.array([x, y]), "Black",
                          1, self.space.y_max)
        self.schedule.add(barrier)

    #################################################################
    ## RUNNING SIMULATION FUNCTIONS
    #################################################################

    @property
    def learn_agent(self):
        return self.cars[0]

    def step(self):
        self.datacollector.collect(self)
        # First loop through and have all cars choose an action
        # before the action is actually propagated forward
        for car in self.cars:
            car.choose_action()

        # Propagate forward one step based on chosen actions
        self.schedule.step()

    def reward(self, agent):
        steering_cost = agent.steer * -200
        acceleration_cost = agent.accel * -100
        speed_reward = 0
        if (agent.speed > agent.target_speed * 1.1):
            speed_reward = -1000
        elif (agent.speed > agent.target_speed * 1.05):
            speed_reward = 200
        elif (agent.speed > agent.target_speed * 0.90):
            speed_reward = 600
        else:
            speed_reward = 100
        # penalizes collisoins from the front more
        collision_cost = agent.collided * -50000

        return speed_reward + acceleration_cost + steering_cost + collision_cost
