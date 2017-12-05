from model import ChaosModel, get_rewards_sum
from mesa.batchrunner import BatchRunner

import numpy as np
import argparse

from matplotlib import pyplot as plt
import os
import pickle
import random

from collections import deque

BATCH_SIZE = 32


class DeepQBatchRunner(BatchRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0       # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def run_model(self, model):
        agent = model.learn_agent
        agent.epsilon = self.epsilon
        for _ in range(4):
            agent.update_state_grid()
        state = np.copy(agent.current_state)
        while model.running and model.schedule.steps < self.max_steps:
            if model.schedule.steps >= 4:
                state = np.copy(agent.current_state)
            model.step()
            if model.schedule.steps >= 4:
                next_state = np.copy(agent.current_state)
                self.memory.append((state, agent.action, agent.reward,
                                    next_state, model.running))
            if len(self.memory) > BATCH_SIZE:
                agent.replay(self.memory, BATCH_SIZE)
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
        agent.save()


def do_plot(out_file, out_file_count, rewards, Q, N):
    curr_out_file = "{}{:04d}".format(out_file, out_file_count)
    pickle.dump([Q, N, rewards], open(curr_out_file, 'wb'))
    out_file_count += 1
    plt.plot(-np.log(-rewards))
    plt.draw()
    plt.pause(0.001)

def train_qlearn(args):
    agent_type = "Q Learn"

    rewards = np.array(())

    if args.load_pickle_file is None:
        Q = None
        N = None
    else:
        Q_logs = pickle.load(open(args.load_pickle_file, 'rb'))
        Q = Q_logs[0]
        N = Q_logs[1]
        rewards = Q_logs[2]

    plt.axis()
    plt.ion()
    plt.show()
    plt.xlabel('Iteration')
    plt.ylabel('reward')

    out_file_count = 0
    for i  in range(args.num_episodes):
        n = args.num_adversaries
        model = ChaosModel(agent_type, 
                           num_adversaries=n, 
                           road_width=60, 
                           Q=Q, N=N)

        for j in range(args.num_steps_per_episode):
            model.step()
        
        Q = model.learn_agent.Q
        N = model.learn_agent.N

        curr = get_rewards_sum(model)
        rewards = np.append(rewards, curr)
        print("Curr reward: {:8.0f}, Max reward: {:8.0f}, Max Q: {:8.0f}, Min Q: {:8.0f}".format(
            curr, rewards.max(), Q.max(), Q.min()))

        if rewards.size % args.plot_freq == 0:
            do_plot(args.out_file, out_file_count, rewards, Q, N)

    rewards = np.array(rewards)
    print("average reward: {:.0f}".format(np.mean(rewards)))

    do_plot(args.out_file, out_file_count, rewards, Q, N)
    plt.savefig('reward_vs_iters.png', dpi=400)
    print(Q)
    input("press Enter to continue...")

def train_deepq(args):
    fixed_params = {"agent_type": "Deep Q Learn",
                    "frozen": False, "epsilon": 1.0,
                    "episode_duration": 1000}
    variable_params = {"num_adversaries": range(10, 11, 1)}
    batch_run = DeepQBatchRunner(ChaosModel,
                                 fixed_parameters=fixed_params,
                                 variable_parameters=variable_params,
                                 iterations=args.num_episodes,
                                 max_steps=500)
    batch_run.run_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument("-e", "--num_episodes", type=int, default=1,
                        help="number of episodes to train for")
    parser.add_argument("-n", "--num_adversaries", type=int, default=1,
                        help="number of adversaries to add")
    parser.add_argument("-t", "--num_steps_per_episode", type=int, default=50,
                        help="number of steps per episode")
    parser.add_argument("-d", "--deepq", action="store_true",
                        help="train with Deep Q Learning agent")
    parser.add_argument('-o', '--out_file', type=str,default = 'train.p',
                        help='Name of file to save Q logs')
    parser.add_argument('--load_pickle_file','-l', type=str,default = None,
                        help='Name of file to save Q logs')
    parser.add_argument('-p', '--plot_freq', type=int,default = 100,
                        help='How often to plot')
    args = parser.parse_args()
    if args.deepq:
        train_deepq(args)
    else:
        train_qlearn(args)
