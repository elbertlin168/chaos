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
                 num_adversaries=8, road_width=60, 
                 episode_duration=200,
                 Q=None, N=None):
        '''
        '''
        self.num_adversaries = num_adversaries
        self.road_width = road_width
        self.episode_duration = episode_duration
        self.schedule = RandomActivation(self)

        self.space = ContinuousSpace(canvas_size, canvas_size, True)
        self.cars = []
        self.agent = []

        self.make_agents(canvas_size, agent_type, Q, N)
        self.running = True

        self.step_count = 0

        self.datacollector = DataCollector(
            model_reporters={"Agent rewards sum": get_rewards_sum})


    def agent_start_state(self):
        pos = np.array((self.space.x_max/2, self.space.y_max-1))
        target_speed = 10
        speed = target_speed
        heading = np.radians(-90)
        return (pos, target_speed, speed, heading)

    def reset(self):

        (pos, target_speed, speed, heading) = self.agent_start_state()

        self.agent.speed = speed
        self.agent.heading = heading
        self.space.place_agent(self.agent, pos)
        self.step_count = 0

    def make_agents(self, canvas_size, agent_type, Q, N):
        '''
        '''

        # Qcar
        (pos, target_speed, speed, heading) = self.agent_start_state()
        color = "Black"
        car_width = 6
        car_length = 12
        if agent_type == 'Basic':
            agent = Car(0, self, pos, speed, heading, self.road_width, 
                color, target_speed, car_length=car_length, car_width=car_width)
        elif agent_type == 'Q Learn':
            agent = QCar(0, self, pos, speed, heading, self.road_width, 
                color, target_speed, car_length=car_length, car_width=car_width,
                Q=Q, N=N)
        # elif agent_type == 'Deep Q Learn':
        #     print(agent_type)
        #     agent = DeepQCar(0, self, pos, speed, heading, self.road_width, 
        #         color, target_speed, car_length=car_length, car_width=car_width)

        self.agent = agent
        self.cars.append(agent)
        self.space.place_agent(agent, pos)
        self.schedule.add(agent)


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
                color = "Gray"

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
        self.step_count += 1
        if self.step_count > self.episode_duration:
            self.reset()


        self.datacollector.collect(self)
        # First loop through and have all cars choose an action
        # before the action is actually propagated forward
        for car in self.cars:
            car.choose_action()

        # Propagate forward one step based on chosen actions
        self.schedule.step()    

