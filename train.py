from model import ChaosModel, get_rewards_sum

import numpy as np
import argparse
from matplotlib import pyplot as plt
import os
import pickle
import random


def do_plot(out_file, out_file_count, rewards, Q, N):
    curr_out_file = "{}{:04d}".format(out_file, out_file_count)
    pickle.dump([Q, N, rewards], open(curr_out_file, 'wb'))
    out_file_count += 1
    plt.plot(-np.log(-rewards))
    plt.draw()
    plt.pause(0.001)

def main(args):
    agent_type = "Deep Q Learn" if args.deepq else "Q Learn"

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
        n = random.randint(1,100) if args.deepq else args.num_adversaries
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
    parser.add_argument('-l', '--load_pickle_file', type=str,default = None,
                        help='Name of file to save Q logs')
    parser.add_argument('-p', '--plot_freq', type=int,default = 100,
                    help='How often to plot')
    main(parser.parse_args())
