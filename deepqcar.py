import numpy as np
import car

class DeepQCar(Car):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def choose_action(self):
		super().choose_action()
