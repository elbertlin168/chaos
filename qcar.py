import numpy as np
from car import Car

class QCar(Car):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def choose_action(self):
		super().choose_action()

		for neighbor in self.get_neighbors():
			pos = neighbor.pos
			vel = neighbor.vel_components()
			nid = neighbor.unique_id
			print("{} to {}, pos: ({:.2f},{:.2f}) , vel: ({:.2f},{:.2f})".format(
				self.unique_id, nid, pos[0], pos[1], vel[0], vel[1]))
