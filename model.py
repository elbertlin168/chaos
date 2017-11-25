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
            x = random.random() * self.space.x_max
            y = random.random() * self.space.y_max
            pos = np.array((x, y))

            # Random speed
            speed = random.random() * 5

            # Random target speed
            target_speed = random.random() * 5 + 10

            # Random heading
            heading = np.radians(random.random()*360 - 180)

            if i == 0:
                pos = np.array((250,250))
                speed = 0
                target_speed = 0
                heading = 0
            else:
                pos = np.array((250, 490))
                speed = 3
                target_speed = 3
                heading = np.radians(-90)

            # Initialize car
            car = Car(i, self, pos, speed, heading, target_speed=target_speed)
            self.cars.append(car)
            self.space.place_agent(car, pos)
            self.schedule.add(car)

    def step(self):
        for car in self.cars:
            car.choose_action()

        self.schedule.step()


