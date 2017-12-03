from model import ChaosModel, get_rewards_sum

import numpy as np

rewards = []
for i  in range(50):
    model = ChaosModel(canvas_size=500, num_adversaries=1, road_width=60)
    start_y = model.agent.pos[1]
    prev_y = start_y
    # count = 0
    for j in range(60):
        model.step()
        y = model.agent.pos[1]
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