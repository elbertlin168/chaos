from model import ChaosModel, get_rewards_sum

import numpy as np
import argparse

NUM_EPISODES = 3 #50
NUM_STEPS_PER_EPISODE = 60

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    
    parser.add_argument(
        '--outfile',
        type=str,
        default = '',
        help='name of file to write s, a, r, sp'
    )

    args = parser.parse_args()

    rewards = []
    for i  in range(NUM_EPISODES):
        model = ChaosModel(canvas_size=500, num_adversaries=3, road_width=60, out_file=args.outfile)
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
        curr = model.get_rewards_sum()
        rewards.append(curr)
        print("{:.0f}".format(curr))

    rewards = np.array(rewards)
    print("average reward: {:.0f}".format(np.mean(rewards)))