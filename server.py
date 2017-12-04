from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule

from model import ChaosModel
from SimpleContinuousModule import SimpleCanvas

from barrier import Barrier

canvas_size = 500

def chaos_draw(agent):
    if type(agent) is Barrier:
        w = agent.width / canvas_size
        h = agent.length / canvas_size
    else:
        w = agent.car_width / canvas_size
        h = agent.car_length / canvas_size
    color = agent.color
    return {"Shape": "rect", "w": w, "h": h, "Filled": "true", "Color": color}

n_slider = UserSettableParameter('slider', "Number of adversaries", 8, 1, 20, 1)
w_slider = UserSettableParameter('slider', "Road width", 60, 10, 500, 10)
a_choice = UserSettableParameter('choice', "Learning agent", "Basic",
                                 choices=["Basic", "Q Learn", "Deep Q Learn"])

chaos_canvas = SimpleCanvas(chaos_draw, canvas_size, canvas_size)
model_params = {
    "canvas_size": canvas_size,
    "num_adversaries": n_slider,
    "road_width": w_slider,
    "agent_type": a_choice
}

chart = ChartModule([{"Label": "Agent rewards sum",
                      "Color": "Black"}],
                    data_collector_name='datacollector')
server = ModularServer(ChaosModel, 
                       [chaos_canvas, chart], 
                       "Chaos", model_params)
