import random
import numpy as np

from mesa import Model
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation

from car import Car


ROAD_WIDTH = 40

def rand_min_max(a, b):
    spread = b - a
    return random.random()*spread + a

def rand_center_spread(center, spread):
    a = center - spread/2
    return random.random()*spread + a

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
            # x = random.random() * self.space.x_max/16 + self.space.x_max/2 - self.space.x_max/32
            x = rand_center_spread(self.space.x_max/2, ROAD_WIDTH)
            # y = random.random() * self.space.y_max
            y = self.space.y_max*i/self.num_adversaries
            pos = np.array((x, y))

            # Random speed
            speed = random.random() * 0

            # Random target speed
            target_speed = rand_min_max(3, 8)

            # Random heading
            # heading = np.radians(random.random()*20 - 90 - 10)
            heading = np.radians(rand_center_spread(-90, 20))

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


            # if i == 0:
            #     pos = np.array((250,450))
            #     speed = 15
            #     target_speed = 15
            #     heading = np.radians(-90)

            # Initialize car
            car = Car(i, self, pos, speed, heading, target_speed=target_speed, road_width = ROAD_WIDTH)
            self.cars.append(car)
            self.space.place_agent(car, pos)
            self.schedule.add(car)

            # while (car.collision_look_ahead(self, [0], [0])):
            #     x = random.random() * self.space.x_max/16 + self.space.x_max/2 - self.space.x_max/32
            #     y = random.random() * self.space.y_max
            #     pos = np.array((x, y))


    def step(self):
        for car in self.cars:
            car.choose_action()

        self.schedule.step()


