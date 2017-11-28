from mesa.visualization.ModularVisualization import ModularServer

from model import ChaosModel
from SimpleContinuousModule import SimpleCanvas

canvas_size = 500

def chaos_draw(agent):
	w = agent.car_width/canvas_size
	h = agent.car_length/canvas_size
	color = agent.color
	return {"Shape": "rect", "w": w, "h": h, "Filled": "true", "Color": color}

chaos_canvas = SimpleCanvas(chaos_draw, canvas_size, canvas_size)
model_params = {
    "canvas_size": canvas_size,
    "num_adversaries": 8,
    "road_width": 60,
}

server = ModularServer(ChaosModel, [chaos_canvas], "Chaos", model_params)
