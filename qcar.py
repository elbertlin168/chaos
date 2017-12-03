import numpy as np
from car import Car
import pandas
import random

EPSILON = .5

def stup(t):
    return "({:.1f}, {:.1f})".format(t[0], t[1])

class Bin():
    def __init__(self, max_val, num_bins, min_val=None):
        '''
        A Bin object has fields num_bins and bin_edges
        The bin_edges are formed by evenly spacing from 
        min_val to max_val with extra bins for -inf to 
        min_val and max_val to inf. 

        If min_val is not given then min_val = -max_val
        '''

        max_val = max_val
        num_bins = num_bins

        if min_val is None:
            min_val = -max_val
        min_val = min_val

        bin_edges = np.linspace(min_val, max_val, num=num_bins-1)
        inf = float("inf")

        bin_edges = np.insert(bin_edges, 0, -inf)
        bin_edges = np.append(bin_edges, inf)

        self.num_bins = num_bins
        self.bin_edges = bin_edges

class Discretize():
    def __init__(self):
        self.bin_dict = {}

    def add_bins(self, name, bins):
        self.bin_dict[name] = bins

    def discretize_var(self, continuous, name):
        '''
        Uses pandas.cut to get discretized version of 
        continuous. 
        Looks up the bin_edges in self.bin_dict[name]
        '''

        curr_bin = pandas.cut([continuous], bins=self.bin_dict[name].bin_edges, labels=False)
        curr_bin = curr_bin[0]
        # print("val: {:.2f}, bins: {}, bin: {}".format(
            # continuous, self.bin_dict[name].bin_edges, curr_bin))
        return curr_bin

    def discretize_1d(self, L):
        '''
        L is a list of tuples (continuous, name)
        containing the continuous value and name of each
        variable

        For example L[0] = (relpos[0], 'relposx')

        This function discretizes the continuous values and
        then combines the discrete values into a single state
        using np.ravel_multi_index and the bin sizes
        '''

        # For each continuous value in L convert to a discrete
        # value and get the num_bins for that variable
        discretes = []
        sizes = []
        for continuous, name in L:
            discretes.append(self.discretize_var(continuous, name))
            sizes.append(self.bin_dict[name].num_bins)


        # Convert n-dimensional state representation to 1 dimension
        discrete_1d = np.ravel_multi_index(discretes, tuple(sizes))

        # print("nd: {}, size: {}, 1d: {}".format(
            # discretes, sizes, discrete_1d))

        return discrete_1d

class QCar(Car):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Discretization
        self.state_bins = Discretize()
        self.state_bins.add_bins('relposx',     Bin(self.road_width, 10))
        self.state_bins.add_bins('relposy',     Bin(self.model.space.y_max, 20))
        self.state_bins.add_bins('relvx',       Bin(1, 5))
        self.state_bins.add_bins('relvy',       Bin(2*self.target_speed, 5))
        self.state_bins.add_bins('heading_err', Bin(np.radians(10), 5))
        self.state_bins.add_bins('speed_err',   Bin(0.5*self.target_speed, 5))

        self.action_bins = Discretize()
        self.action_bins.add_bins('steer', Bin(self.steer_mag/2, 3))
        self.action_bins.add_bins('accel', Bin(self.accel_mag/2, 3))

        # Initialize
        self.rewards_sum = 0

        self.states = []
        for neighbor in self.get_neighbors():
            self.states.append(-1)

    def choose_action(self):
        super().choose_action()

        # Change each action to random selection with probability EPSILON 
        steer_actions = [self.turn_right(), self.turn_left(), self.go_straight()]
        accel_actions = [self.maintain_speed(), self.accelerate(), self.brake()]

        if random.random() > EPSILON:
            # print('random steer')
            self.steer = random.choice(steer_actions)

        if random.random() > EPSILON:
            # print('random accel')
            self.accel = random.choice(accel_actions)

    def step(self):
        super().step()

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
                with open("train.out", 'a') as f:
                    f.write("{} {:6.0f}, {:6.0f}, {:10.2f}, {:6.0f}\n".format(
                    i, prev_state, action, reward, state))

    def get_action(self):
        states = []
        states.append((self.steer, 'steer'))
        states.append((self.accel, 'accel'))
        return self.action_bins.discretize_1d(states)

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
            
        