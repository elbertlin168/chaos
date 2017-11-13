from mesa.visualization.ModularVisualization import ModularServer

from model import ChaosModel
from SimpleContinuousModule import SimpleCanvas


def chaos_draw(agent):
    return {"Shape": "circle", "r": 2, "Filled": "true", "Color": "Red"}

chaos_canvas = SimpleCanvas(chaos_draw, 500, 500)
model_params = {
    "lanes": 5,
    "num_adversaries": 50,
    "max_speed": 30,
}

server = ModularServer(ChaosModel, [chaos_canvas], "Chaos", model_params)