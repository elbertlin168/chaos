from model import ChaosModel

# server.launch()
model = ChaosModel(canvas_size=500, num_adversaries=8, road_width=60)
for i in range(100):
    model.step()