import numpy as np
from car import Car

def stup(t):
	return "({:.1f}, {:.1f})".format(t[0], t[1])

class QCar(Car):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def choose_action(self):
		super().choose_action()

		# print("reward: {}".format(self.cum_reward))
		print("Self pos: ({:.2f},{:.2f}) , vel: ({:.2f},{:.2f})".format(
			self.pos[0], self.pos[1], 
			self.vel_components()[0], self.vel_components()[1]))

		for neighbor in self.get_neighbors():
			pos = neighbor.pos
			vel = neighbor.vel_components()
			nid = neighbor.unique_id
			relpos = pos - self.pos
			relv = vel - self.vel_components()
			print("{} to {}, pos: {}, vel: {}, relpos: {}, relv: {}".format(
				self.unique_id, nid, stup(pos), stup(vel), 
				stup(relpos), stup(relv)))

