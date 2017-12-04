from model import ChaosModel, get_rewards_sum

import numpy as np
import argparse

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

    args = parser.parse_args()

    rewards = []
    for i  in range(args.num_episodes):
        model = ChaosModel(args.agent_type, 
                           canvas_size=500, 
                           num_adversaries=args.num_adversaries, 
                           road_width=60)
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

        # print(count)
        curr = model.agent.rewards_sum
        rewards.append(curr)
        print("{:.0f}".format(curr))

    rewards = np.array(rewards)
    print("average reward: {:.0f}".format(np.mean(rewards)))