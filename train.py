from model import ChaosModel, get_rewards_sum

import numpy as np
import sys, argparse
import random


def main(deepq):
    agent_type = "Deep Q Learn" if deepq else "Q Learn"
    rewards = []
    for i  in range(50):
        n = random.randint(1,100) if deepq else 1
        model = ChaosModel(agent_type, num_adversaries=n)
        start_y = model.learn_agent.pos[1]
        prev_y = start_y
        # count = 0
        for j in range(60):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--deepq", action="store_true",
                        help="train with Deep Q Learning agent")
    args = parser.parse_args()
    main(args.deepq)
