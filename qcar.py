import numpy as np
from car import Car
import pandas
import random

def stup(t):
    return "({:.1f}, {:.1f})".format(t[0], t[1])

def make_bins(lim, num_bins):
    bins = np.linspace(-lim, lim, num=num_bins-1)
    inf = float("inf")

    bins = np.insert(bins, 0, -inf)
    bins = np.append(bins, inf)
    return bins

class QCar(Car):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # discretization
        # relposx_bins = np.linspace(-self.road_width, self.road_width, num=2)
        # relposy_bins = np.linspace(-self.model.space.y_max, self.model.space.y_max, num=2)
        # relvx_bins = np.linspace(-1, 1, num=2)
        # relvy_bins = np.linspace(-2*self.target_speed, 2*self.target_speed, num=2)

        relposx_bins = make_bins(self.road_width, 10)
        relposy_bins = make_bins(self.model.space.y_max, 20)
        relvx_bins = make_bins(1, 5)
        relvy_bins = make_bins(2*self.target_speed, 5)

        self.state_bins_list = [relposx_bins, relposy_bins, relvx_bins, relvy_bins]
        self.state_bins_size_list = []
        # self.bins_list = []
        # inf = float("inf")
        for bins in self.state_bins_list:
            # curr = np.insert(bins, 0, -inf)
            # curr = np.append(curr, inf)
            # self.bins_list.append(curr)
            self.state_bins_size_list.append(len(bins)-1)


        steer_bins = make_bins(self.steer_mag/2, 3)
        accel_bins = make_bins(self.accel_mag/2, 3)


        self.action_bins_list = [steer_bins, accel_bins]
        self.action_bins_size_list = []
        # self.bins_list = []
        # inf = float("inf")
        for bins in self.action_bins_list:
            # curr = np.insert(bins, 0, -inf)
            # curr = np.append(curr, inf)
            # self.bins_list.append(curr)
            self.action_bins_size_list.append(len(bins)-1)


        # the sum of the rewards this agent recieved
        self.rewards_sum = 0

        self.state = -1

    def choose_action(self):
        super().choose_action()

        steer_actions = [self.turn_right(), self.turn_left(), self.go_straight()]

        accel_actions = [self.maintain_speed(), self.accelerate(), self.brake()]

        if random.random() > 0.1:
            self.steer = random.choice(steer_actions)

        if random.random() > 0.1:
            self.accel = random.choice(accel_actions)

    def step(self):
        super().step()

        reward = self.reward()

        self.rewards_sum += reward

        prev_state = self.state
        self.state = self.get_state()
        action = self.get_action()

        if prev_state >= 0:
            with open("train.out", 'a') as f:
                f.write("{:6.0f}, {:6.0f}, {:10.2f}, {:6.0f}\n".format(
                prev_state, action, reward, self.state))
        # print("state: {}, action: {}, reward: {:.1f}, sum: {:.1f}".format(
            # state, action, reward, self.rewards_sum))

    def get_action(self):
        action_nd = []
        # print("steer: {:.5f}, accel: {:.5f}".format(self.steer, self.accel))
        for bins, val in zip(self.action_bins_list, [self.steer, self.accel]):
            # print("val: {}".format(val))
            # print("bins: {}".format(bins))
            q = pandas.cut([val], bins=bins, labels=False)
            # print("q {}".format(q))
            action_nd.append(q)

        # print("action_nd: {}".format(action_nd))

        # print("bins size list {}".format(tuple(self.action_bins_size_list)))
        # print("bins list {}".format(self.action_bins_list)) 
        action = np.ravel_multi_index(action_nd, tuple(self.action_bins_size_list))
        # print("action: {}".format(action))

        return action[0]

    def get_state(self):

        # print("reward: {}".format(self.cum_reward))
        # print("Self pos: ({:.2f},{:.2f}) , vel: ({:.2f},{:.2f})".format(
            # self.pos[0], self.pos[1], 
            # self.vel_components()[0], self.vel_components()[1]))

        for neighbor in self.get_neighbors():
            pos = neighbor.pos
            vel = neighbor.vel_components()
            nid = neighbor.unique_id
            relpos = pos - self.pos
            relv = vel - self.vel_components()

            state_vars = [relpos[0], relpos[1], relv[0], relv[1]]

            state_nd = []
            for bins, val in zip(self.state_bins_list, state_vars):
                # print("val: {}".format(val))
                # print("bins: {}".format(bins))
                q = pandas.cut([val], bins=bins, labels=False)
                # print("q {}".format(q))
                state_nd.append(q)


            # print("{} to {}, pos: {}, vel: {}, relpos: {}, relv: {}".format(
                # self.unique_id, nid, stup(pos), stup(vel), 
                # stup(relpos), stup(relv)))


            # print("bins size list {}".format(tuple(self.bins_size_list)))
            # print("bins list {}".format(self.bins_list))  


            # print("state_nd {}".format(state_nd))
            state = np.ravel_multi_index(state_nd, tuple(self.state_bins_size_list))
            # state_nd2 = np.unravel_index(state, tuple(self.bins_size_list))

            # print("state_1d {}".format(state_1d))
            # print("state_nd2 {}".format(state_nd2))

            return state[0]
            
        