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
    return model.rewards_sum

def get_rewards_sum_log(model):
    return -np.log(-get_rewards_sum(model))

class ChaosModel(Model):
    '''
    The stochastic highway simulation model
    '''

    #################################################################
    ## INITIALIZATION FUNCTIONS
    #################################################################

    def __init__(self, agent_type, width=500, height=500, 
                 num_adversaries=8, road_width=60,
                 episode_duration=100,
                 Q=None, N=None):
        self.num_adversaries = num_adversaries
        self.road_width = road_width
        self.episode_duration = episode_duration
        self.schedule = RandomActivation(self)

        self.space = ContinuousSpace(width, height, True)
        self.cars = []
        self.FROZEN = True # Adds one frozen adversary if True

        self.make_agents(AgentType(agent_type), Q, N)
        self.running = True

        self.curr_reward = 0

        self.reset()

        self.datacollector = DataCollector(
            model_reporters={"Agent rewards sum": get_rewards_sum_log})

    def agent_start_state(self):
        pos = np.array((self.space.x_max/2, self.space.y_max/2+100))

        target_speed = 10
        speed = target_speed
        heading = np.radians(-90)
        return (pos, target_speed, speed, heading)

    def reset(self):

        (pos, target_speed, speed, heading) = self.agent_start_state()
        # speed = util.rand_min_max(target_speed*.8, target_speed*1.2)
        # speed = util.rand_min_max(6,14)
        # speed = 0
        # heading = util.rand_min_max(np.radians(-89),np.radians(-91))
        # heading = np.radians(-85)
        self.learn_agent.speed = speed
        self.learn_agent.heading = heading
        self.space.place_agent(self.learn_agent, pos)
        self.step_count = 0
        self.rewards_sum = -1

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

    def make_learn_agent(self, agent_type, unique_id, Q, N):
        
        (pos, target_speed, speed, heading) = self.agent_start_state()
        width = 6
        length = 12
        color = "Black"

        if agent_type == AgentType.QLEARN:
            return QCar(unique_id, self, pos, speed, heading, color, 
                        target_speed=target_speed, length=length, 
                        width=width, Q=Q, N=N)

        elif agent_type == AgentType.DEEPQ:
            car = DeepQCar

        else: # agent_type == AgentType.BASIC
            car = Car

        return car(unique_id, self, pos, speed, heading, color,
                  target_speed=target_speed, width=width, length=length)

    def make_frozen(self, unique_id):
        pos = np.array((self.space.x_max/2, self.space.y_max/2))
        return Car(unique_id, self, pos, 0, np.radians(-90), "Indigo",
                  target_speed=0, width=6, length=12)

    def make_agents(self, agent_type, Q, N):
        '''
        Add all agents to model space
        '''
        # Car agents
        for i in range(0, self.num_adversaries + 1):
            if i == 0:
                car = self.make_learn_agent(agent_type, i, Q, N)
            elif self.FROZEN:
                car = self.make_frozen(i)
                self.FROZEN = False
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

        self.curr_reward = self.reward()
        self.rewards_sum += self.curr_reward

    def reward(self):
        agent = self.learn_agent
        # heading_cost = np.abs(agent.heading - agent.target_heading) * -2
        # steering_cost = np.abs(agent.steer) * -2
        # acceleration_cost = np.abs(agent.accel) * -1
        vy = -agent.speed * np.sin(agent.heading)
        speed_reward = 0
        if (vy > agent.target_speed * 1.1):
            speed_reward = -20
        elif (vy > agent.target_speed * 1.05):
            speed_reward = -2
        elif (vy > agent.target_speed * 0.90):
            speed_reward = 0
        else:
            speed_reward = -3
        # penalizes collisoins from the front more
        collision_cost = agent.collided * -500

        # Alternate speed reward
        speed_reward = np.abs(vy - agent.target_speed) * -1
        if speed_reward > -0.5:
            speed_reward = 0
        elif speed_reward < -agent.target_speed/2:
            speed_reward = -20

        # Try to reward position
        # pos_cost = 0
        # if agent.pos[1] < self.space.y_max/2 - 70 or vy < agent.target_speed/2:
        #     pos_cost = -10 
        # print(pos_reward)

        # print("speed reward {}, accel cost {}, steer cost {}, collision cost {}".format(
            # speed_reward, acceleration_cost, steering_cost, collision_cost))
        # return speed_reward + acceleration_cost + steering_cost + \
               # collision_cost + heading_cost
        # print(speed_reward, collision_cost, heading_cost)

        # print(vy, agent.target_speed, speed_reward, collision_cost, a, agent.steer, agent.accel)

        return speed_reward + collision_cost