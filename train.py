from model import ChaosModel

# server.launch()
model = ChaosModel(canvas_size=500, num_adversaries=10, road_width=60)
start_y = model.agent.pos[1]
prev_y = start_y
while True:
    model.step()
    y = model.agent.pos[1]
    # print("pos y: {:.1f}".format(y))
    if prev_y - y < 0:
        break
    prev_y = y