from mesa.visualization.ModularVisualization import ModularServer

from model import ChaosModel
from SimpleContinuousModule import SimpleCanvas


def chaos_draw(agent):
    return {"Shape": "rect", "w": 0.01, "h": 0.05, "Filled": "true", "Color": "Red"}

canvas_size = 500
chaos_canvas = SimpleCanvas(chaos_draw, canvas_size, canvas_size)
model_params = {
    "canvas_size": canvas_size,
    "num_adversaries": 8,
    "road_width": 40,
}

server = ModularServer(ChaosModel, [chaos_canvas], "Chaos", model_params)
