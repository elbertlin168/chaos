from model import ChaosModel, get_rewards_sum

import numpy as np
import argparse
import random

NUM_STEPS_PER_EPISODE = 60


def main(args):
    agent_type = "Deep Q Learn" if args.deepq else "Q Learn"
    rewards = []
    for i in range(args.num_episodes):
        n = random.randint(1,100) if args.deepq else args.num_adversaries
        model = ChaosModel(agent_type, num_adversaries=n)
        start_y = model.learn_agent.pos[1]
        prev_y = start_y
        # count = 0
        for j in range(NUM_STEPS_PER_EPISODE):
            model.step()
            y = model.learn_agent.pos[1]
            # print("pos y: {:.1f}".format(y))
            # if prev_y - y < 0:
                # break
            prev_y = y
            # count = count + 1

        # print(count)
        curr = get_rewards_sum(model)
        rewards.append(curr)
        print("{:.0f}".format(curr))

    rewards = np.array(rewards)
    print("average reward: {:.0f}".format(np.mean(rewards)))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument("-e", "--num_episodes", type=int, default=1,
                        help="number of episodes to train for")
    parser.add_argument("-n", "--num_adversaries", type=int, default=1,
                        help="number of adversaries to add")
    parser.add_argument("-d", "--deepq", action="store_true",
                        help="train with Deep Q Learning agent")
    main(parser.parse_args())
