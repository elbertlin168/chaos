from model import ChaosModel, get_rewards_sum

import numpy as np
import argparse
from matplotlib import pyplot as plt
import os
import pickle

NUM_STEPS_PER_EPISODE = 60

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')

    parser.add_argument(
         '--num_episodes', '-e',
        type=int,
        default = 1,
        help='# of episodes'
    )

    parser.add_argument(
        '--num_adversaries', '-n', 
        type=int,
        default = 1,
        help='num adversaries'
    )

    parser.add_argument(
        '--agent_type','-t', 
        type=str,
        default = 'Q Learn',
        help='Agent type')

    parser.add_argument(
        '--out_file','-o', 
        type=str,
        default = 'train.p',
        help='Name of file to save Q logs')

    parser.add_argument(
        '--load_pickle_file','-l', 
        type=str,
        default = None,
        help='Name of file to save Q logs')

    parser.add_argument(
        '--showplot',
        type=int,
        default = 0,
        help='Flag to call pyplot.show(). 1 to show plot otherwise will not show'
    )

    args = parser.parse_args()

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
        model = ChaosModel(args.agent_type, 
                           canvas_size=500, 
                           num_adversaries=args.num_adversaries, 
                           road_width=60, 
                           Q=Q, N=N)
        start_y = model.agent.pos[1]
        prev_y = start_y
        # count = 0
        for j in range(NUM_STEPS_PER_EPISODE):
            model.step()
            y = model.agent.pos[1]
            # print("pos y: {:.1f}".format(y))
            # if prev_y - y < 0:
                # break
            prev_y = y
            # count = count + 1


        Q = model.agent.Q
        N = model.agent.N

        # print(count)
        curr = model.agent.rewards_sum
        rewards = np.append(rewards, curr)
        print("Curr reward: {:8.0f}, Max reward: {:8.0f}, Max Q: {:8.0f}, Min Q: {:8.0f}".format(
            curr, rewards.max(), Q.max(), Q.min()))

        if rewards.size % 100 == 0:
            curr_out_file = "{}{:04d}".format(args.out_file, out_file_count)
            pickle.dump([Q, N, rewards], open(curr_out_file, 'wb'))
            out_file_count += 1
            plt.plot(-np.log(-rewards))
            plt.draw()
            plt.pause(0.001)



    rewards = np.array(rewards)
    print("average reward: {:.0f}".format(np.mean(rewards)))

    curr_out_file = "{}{:04d}".format(args.out_file, out_file_count)
    pickle.dump([Q, N, rewards], open(curr_out_file, 'wb'))
    out_file_count += 1
    plt.plot(-np.log(-rewards))
    plt.draw()
    plt.pause(0.001)

    plt.savefig('reward_vs_iters.png', dpi=400)
    input("press Enter to continue...")
