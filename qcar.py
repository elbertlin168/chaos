import numpy as np
from car import Car
import random
from Discretize import *
from settings import *

EPSILON = 0.9

class QCar(Car):

    def __init__(self,
                 unique_id,
                 model,
                 pos,
                 speed,
                 heading,
                 road_width,
                 color,
                 target_speed,
                 target_heading = TARGET_HEADING,
                 speed_margin=SPEED_MARGIN,
                 heading_margin=HEADING_MARGIN,
                 accel_mag=ACCEL_MAG,
                 steer_mag=STEER_MAG,
                 accuracy=ACCURACY,
                 safety_margin=SAFETY_MARGIN,
                 car_width=CAR_WIDTH,
                 car_length=CAR_LENGTH,
                 state_size=STATE_SIZE,
                 Q = None, 
                 N = None
                 ):

        super().__init__(
                 unique_id,
                 model,
                 pos,
                 speed,
                 heading,
                 road_width,
                 color,
                 target_speed,
                 target_heading = TARGET_HEADING,
                 speed_margin=SPEED_MARGIN,
                 heading_margin=HEADING_MARGIN,
                 accel_mag=ACCEL_MAG,
                 steer_mag=STEER_MAG,
                 accuracy=ACCURACY,
                 safety_margin=SAFETY_MARGIN,
                 car_width=CAR_WIDTH,
                 car_length=CAR_LENGTH,
                 state_size=STATE_SIZE)

        # Discretization
        self.state_bins = Discretize()
        self.state_bins.add_bins('relposx',     Bin(self.road_width, 1))
        self.state_bins.add_bins('relposy',     Bin(self.model.space.y_max, 1))
        self.state_bins.add_bins('relvx',       Bin(1, 1))
        self.state_bins.add_bins('relvy',       Bin(2*self.target_speed, 1))
        self.state_bins.add_bins('heading_err', Bin(np.radians(10), 1))
        self.state_bins.add_bins('speed_err',   Bin(0.5*self.target_speed, 50))

        self.action_bins = Discretize()
        self.action_bins.add_bins('steer', Bin(self.steer_mag/2, 1))
        self.action_bins.add_bins('accel', Bin(self.accel_mag/2, 3))

        # Initialize
        self.rewards_sum = 0

        self.states = []
        for neighbor in self.get_neighbors():
            self.states.append(-1)

        # self.out_file = out_file
        self.out_file = 'train.out'

        # print(self.state_bins.size(), self.action_bins.size())

        if Q is None:
            Q = np.zeros((self.state_bins.size(), self.action_bins.size()))
        if N is None:
            N = np.zeros((self.state_bins.size(), self.action_bins.size()))
        self.Q = Q
        self.N = N
        # self.alpha = 1
        self.discount = 1

    def choose_action(self):
        # super().choose_action()

        # Print state, action, reward, next state to file
        reward = self.reward()

        self.rewards_sum += reward

        prev_states = self.states
        self.states = self.get_all_states()
        action = self.get_action()

        for i in range(len(prev_states)):
            prev_state = prev_states[i]
            state = self.states[i]
            if prev_state >= 0:
                self.N[prev_state, action] += 1
                alpha = 1/self.N[prev_state, action]
                self.Q[prev_state, action] += alpha*(
                    reward + self.discount * self.Q[state].max() - self.Q[prev_state, action])
                # if self.out_file:
                #     with open(self.out_file, 'a') as f:
                #         f.write("{} {:6.0f}, {:6.0f}, {:10.2f}, {:6.0f}\n".format(
                #         i, prev_state, action, reward, state))

        # action = self.Q[self.states[0]].argmax()
        Qcurr = self.Q[self.states[0]]
        m = Qcurr.max()
        # if m > 0:
            # print("Qmax {}".format(m))
        action = np.random.choice(np.flatnonzero(Qcurr == m))
        (self.steer, self.accel) = self.action_bins.unravel(action, 
                                                            ['steer', 'accel'])

        print(self.Q)


        # Change each action to random selection with probability EPSILON 
        steer_actions = [self.turn_right(), self.turn_left(), self.go_straight()]
        accel_actions = [self.maintain_speed(), self.accelerate(), self.brake()]

        if random.random() > EPSILON:
            # print('random steer')
            self.steer = random.choice(steer_actions)

        if random.random() > EPSILON:
            # print('random accel')
            self.accel = random.choice(accel_actions)

        self.steer = 0

        # super().choose_action()

    def step(self):
        super().step()



    def get_action(self):
        actions = []
        actions.append((self.steer, 'steer'))
        actions.append((self.accel, 'accel'))
        return self.action_bins.discretize_1d(actions)

    def get_all_states(self):

        speed_err = self.speed - self.target_speed
        heading_err = self.heading - self.target_heading

        # There is a state for each neighbor
        all_states = []
        for neighbor in self.get_neighbors():
            pos = neighbor.pos
            vel = neighbor.vel_components()
            nid = neighbor.unique_id
            relpos = pos - self.pos
            relv = vel - self.vel_components()

            states = []
            states.append((relpos[0], 'relposx'))
            states.append((relpos[1], 'relposy'))
            states.append((relv[0], 'relvx'))
            states.append((relv[1], 'relvy'))
            states.append((speed_err, 'speed_err'))
            states.append((heading_err, 'heading_err'))

            all_states.append(self.state_bins.discretize_1d(states))

        return all_states
            
        