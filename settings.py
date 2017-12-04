import numpy as np
from enum import Enum

class AgentType(Enum):
	BASIC = "Basic"
	QLEARN = "Q Learn"
	DEEPQ = "Deep Q Learn"


# Default car movement
TARGET_SPEED = 5
TARGET_HEADING = -np.radians(90)
SPEED_MARGIN = 0.1                 # Unstable if << ACCEL_MAG
HEADING_MARGIN = np.radians(1)     # Unstable if << STEER_MAG
ACCEL_MAG = 0.5
STEER_MAG = np.radians(3)

# Car dimensions
CAR_WIDTH = 10 #5 #12
CAR_LENGTH = 20 #25 #100

# Variables for collision avoidance
ACCURACY = 1
SAFETY_MARGIN = 0.3
LOOKAHEAD = 5
COLLIDE_FRONT = 1
COLLIDE_BACK = 2

# How much to prefer left turns over right turns, with 1 always
# attempting left first and 0 always right first
LEFT_TURN_PREFERENCE = 0.5

# Minimum allowed speed in bicycle model
MIN_SPEED = 0
