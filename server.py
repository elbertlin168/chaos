from mesa.visualization.ModularVisualization import ModularServer

from model import ChaosModel
from SimpleContinuousModule import SimpleCanvas


def chaos_draw(agent):
    return {"Shape": "rect", "w": 0.03, "h": 0.2, "Filled": "true", "Color": "Red"}

lanes = 5
chaos_canvas = SimpleCanvas(chaos_draw, 500, 500)
model_params = {
    "lanes": lanes,
    "num_adversaries": 2,
    "max_speed": 30,
}

server = ModularServer(ChaosModel, [chaos_canvas], "Chaos", model_params)
