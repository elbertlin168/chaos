import numpy as np
from car import Car

class QCar(Car):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def choose_action(self):
		super().choose_action()
