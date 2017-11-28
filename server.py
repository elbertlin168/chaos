from mesa.visualization.ModularVisualization import ModularServer

from model import ChaosModel
from SimpleContinuousModule import SimpleCanvas

canvas_size = 500

def chaos_draw(agent):
    return {"Shape": "rect", "w": agent.car_width/canvas_size, "h": agent.car_length/canvas_size, "Filled": "true", "Color": "Red"}

chaos_canvas = SimpleCanvas(chaos_draw, canvas_size, canvas_size)
model_params = {
    "canvas_size": canvas_size,
    "num_adversaries": 8,
    "road_width": 40,
}

server = ModularServer(ChaosModel, [chaos_canvas], "Chaos", model_params)
