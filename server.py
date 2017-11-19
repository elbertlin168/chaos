from mesa.visualization.ModularVisualization import ModularServer

from model import ChaosModel
from SimpleContinuousModule import SimpleCanvas


def chaos_draw(agent):
    return {"Shape": "rect", "w": 0.01, "h": 0.025, "Filled": "true", "Color": "Red"}

lanes = 25
chaos_canvas = SimpleCanvas(chaos_draw, 500, lanes * 20)
model_params = {
    "lanes": lanes,
    "num_adversaries": 50,
    "max_speed": 30,
}

server = ModularServer(ChaosModel, [chaos_canvas], "Chaos", model_params)
