from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule

from model import ChaosModel
from SimpleContinuousModule import SimpleCanvas

from barrier import Barrier
from settings import AgentType

canvas_size = 500

def chaos_draw(agent):
    w = agent.width / canvas_size
    h = agent.length / canvas_size
    color = agent.color
    return {"Shape": "rect", "w": w, "h": h, "Filled": "true", "Color": color}

n_slider = UserSettableParameter('slider', "Number of adversaries", 8, 1, 20, 1)
w_slider = UserSettableParameter('slider', "Road width", 60, 10, 500, 10)
a_choice = UserSettableParameter('choice', "Learning agent", AgentType.BASIC.value,
                                 choices=[agent.value for agent in AgentType])
e_dur = UserSettableParameter('slider', "Episode Duration", 50, 5, 200, 5)

chaos_canvas = SimpleCanvas(chaos_draw, canvas_size, canvas_size)
model_params = {
    "num_adversaries": n_slider,
    "road_width": w_slider,
    "agent_type": a_choice,
    "episode_duration": e_dur
}

chart = ChartModule([{"Label": "Agent rewards sum",
                      "Color": "Black"}],
                    data_collector_name='datacollector')
server = ModularServer(ChaosModel, 
                       [chaos_canvas, chart], 
                       "Chaos", model_params)
