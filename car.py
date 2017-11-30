import numpy as np
import random
from mesa import Agent
import math
import util


# Desired heading
TARGET_HEADING = -np.radians(90)

# Allowed error on speed.
# Warning: If the margin is too small relative to the
# accel magnitude it will be unstable
SPEED_MARGIN = 0.1

# Allowed error on heading.
# Warning: If the margin is too small relative to the
# steer magnitude, it will be unstable
HEADING_MARGIN = np.radians(1)

# How much accel/brake in each action
ACCEL_MAG = 0.5

# How much steering in each action
STEER_MAG = np.radians(3)

# Accuracy in propagation during collision detection
ACCURACY = 1

# Safety buffer around car dimensions in collision detection
SAFETY_MARGIN = 0.3

# Car dimensions
CAR_WIDTH = 10 #5 #12
CAR_LENGTH = 20 #25 #100

# How far to lookahead for collision avoidance
LOOKAHEAD = 5

# Distance to look for neighbors
# GET_NEIGHBOR_DIST = 100

# Enumerate code for collide front and back
COLLIDE_FRONT = 1
COLLIDE_BACK = 2

# How much to prefer left turns over right turns.
# If 1.0 then left turn will always be attempted first and
# right turn only performed if left results in collision
# If 0.5 then half the time a left turn will be tried first
# and half the time right turn will be tried first
LEFT_TURN_PREFERENCE = 0.5

# Minimum allowed speed in bicycle model
MIN_SPEED = 0

# The dimensions of the grid that make up the state
STATE_SIZE = 5

# Minimum distance from neighbors
# RISK_TOLERANCE = 0.5
# ATTENTION = 0.95

def wrap_angle(angles):
   '''
   Returns angle betwee -pi and pi
   '''
   return ( angles + np.pi) % (2 * np.pi ) - np.pi

class Car(Agent):
    '''
    '''

    def __init__(self,
                 unique_id,
                 model,
                 pos,
                 speed,
                 heading,
                 road_width,
                 color,
                 target_speed,
                 target_heading = TARGET_HEADING,
                 speed_margin=SPEED_MARGIN,
                 heading_margin=HEADING_MARGIN,
                 accel_mag=ACCEL_MAG,
                 steer_mag=STEER_MAG,
                 accuracy=ACCURACY,
                 safety_margin=SAFETY_MARGIN,
                 car_width=CAR_WIDTH,
                 car_length=CAR_LENGTH,
                 state_size=STATE_SIZE
                 # risk_tolerance=RISK_TOLERANCE,
                 # attention=ATTENTION,
                 ):
        '''
        '''
        super().__init__(unique_id, model)

        # Initial position
        self.pos = np.array(pos)

        # Initial speed
        self.speed = speed

        # Initial heading
        self.heading = heading

        # Road width. Sets the boundary
        self.road_width = road_width

        # Set orig color
        self.orig_color = color
        self.color = color

        # Target speed and heading
        self.target_speed = target_speed
        self.target_heading = target_heading

        # Speed and heading margin
        self.speed_margin = speed_margin
        self.heading_margin = heading_margin

        # Steer/accel magnitudes
        self.accel_mag = accel_mag
        self.steer_mag = steer_mag

        # accuracy of state propagation in collision detection
        self.accuracy = accuracy

        # margin for collision avoidance
        self.safety_margin = safety_margin

        # Size of car for collision detection
        self.car_width = car_width
        self.car_length = car_length

        # self.risk_tolerance = risk_tolerance
        # self.attention = attention

        # Initialize the bicycle model
        # These constants can be varied if we want to change
        # how the steering angle effects the turning kinematics
        self.lr = 1
        self.lf = 1

        # Initialize inputs to 0

          # longitudinal acceleration effects speed directly
        self.accel = 0

          # steering angle in radians controls the angular velocity
          # via the bicycle model kinematics
        self.steer = 0

        # the sum of the rewards this agent recieved
        self.rewards_sum = 0

        self.state_size = state_size

    def choose_action(self):
        '''
        Assigns steer and accel values to try and meet target
        heading and speed.
        If a collision is detetected, tries to choose action
        to avoid collision
        '''

        # Reset color
        self.color = self.orig_color

        # Choose accel and steer to aim for target speed and heading
        # Do this action only for the first step in the sequence
        steers = self.heading_control()
        accels = self.speed_control()

        # If a collision is detected
        collision_detection = self.collision_lookahead(steers, accels)
        if (collision_detection == COLLIDE_BACK) or \
        (collision_detection == COLLIDE_FRONT and steers[0] != 0):
            # print("id: {} collision: {}".format(
            #         self.unique_id, collision_detection))

            (steers, accels) = self.avoid_collision(collision_detection)

        # Assign action to object
        self.steer = steers[0]
        self.accel = accels[0]

    def reward(self, collided):
        steering_cost = self.heading * -200
        acceleration_cost = self.accel * -100
        speed_reward = 0
        if (self.speed > self.target_speed * 1.1):
            speed_reward = -1000
        elif (self.speed > self.target_speed * 1.05):
            speed_reward = 200
        elif (self.speed > self.target_speed * 0.90):
            speed_reward = 600
        else:
            speed_reward = 100
        # penalizes collisoins from the front more
        collision_cost = collided * -50000
        return speed_reward + acceleration_cost + steering_cost + collision_cost

    def step(self):
        '''
        Uses chosen actions (steer and accel) to propagate state
        (pos, speed, heading) with the bicycle model.
        '''

        # Propagate state
        (self.speed, self.heading, next_pos) = self.bicycle_model( \
            self.steer, self.accel, self.speed, self.heading, self.pos)

        # Check collision

        collided = self.collision_lookahead(np.array([self.steer]), np.array([self.accel]))

        # Move agent
        self.model.space.move_agent(self, next_pos)

        next_reward = self.reward(collided)

        self.rewards_sum += next_reward
        return next_reward

        # print("id: {} steer: {} accel: {} speed: {} heading {}".format(
        #     self.unique_id, self.steer, self.accel, self.speed,
        #     np.degrees(self.heading)))

    def speed_control(self):
        # Steers and accels represent a sequence of actions over the next
        # N steps. N is controlled by LOOKAHEAD.
        # Initialize these vectors with 0
        accels = np.zeros(LOOKAHEAD)

        speed_error = self.speed - self.target_speed
        if speed_error < -self.speed_margin:
            accels[0] = self.accelerate()
        elif speed_error > self.speed_margin:
            accels[0] = self.brake()
        else:
            accels[0] = self.maintain_speed()

        return accels

    def heading_control(self):
        # Steers and accels represent a sequence of actions over the next
        # N steps. N is controlled by LOOKAHEAD.
        # Initialize these vectors with 0
        steers = np.zeros(LOOKAHEAD)
        accels = np.zeros(LOOKAHEAD)

        heading_error = wrap_angle(self.heading - TARGET_HEADING)

        if heading_error < -self.heading_margin:
            steers[0] = self.turn_left()
        elif heading_error > self.heading_margin:
            steers[0] = self.turn_right()
        else:
            steers[0] = self.go_straight()

        return steers

    def avoid_collision(self, collision_detection):

        # Enumerate heading actions in priority order.
        # In order to not prioritize left or right use a random number
        # to choose which one to try first.
        if random.random() > LEFT_TURN_PREFERENCE:
            steer_actions = [self.turn_left(), self.turn_right(), self.go_straight()]
        else:
            steer_actions = [self.turn_right(), self.turn_left(), self.go_straight()]

        # Enumerate speed actions in priority order
        accel_actions = [self.maintain_speed(), self.accelerate(), self.brake()]


        # Try all combinations of speed and heading actions
        for steer in steer_actions:
            for accel in accel_actions:

                # Steers and accels represent a sequence of actions over the next
                # N steps. N is controlled by LOOKAHEAD.
                # Initialize these vectors with 0
                steers = np.zeros(LOOKAHEAD)
                accels = np.zeros(LOOKAHEAD)

                # Try applying action between 1 and N times
                # in the action sequence
                for i in range(0, LOOKAHEAD):

                    # Add steering and acceleration actions
                    # to the action vectors
                    steers[i] = steer
                    accels[i] = accel

                    # If action sequence no longer results in collision
                    # then return the actions for the first step
                    if self.collision_lookahead(steers, accels) == 0:
                        return (steers, accels)

        # If all possible actions are exhausted then return None
        print('Could not avoid collision')
        return self.resolve_collision(collision_detection)

    def resolve_collision(self, collision_detection):
        # Steers and accels represent a sequence of actions over the next
        # N steps. N is controlled by LOOKAHEAD.
        # Initialize these vectors with 0
        steers = np.zeros(LOOKAHEAD)
        accels = np.zeros(LOOKAHEAD)

        # When collision is unavoidable just go straight
        # And accel or decel based on whether you are
        # the front or back car in the collision
        steers[0] = self.go_straight()
        if collision_detection == COLLIDE_FRONT:
            accels[0] = self.accelerate()
        if collision_detection == COLLIDE_BACK:
            accels[0] = self.brake()

        # Indicate collision
        self.color = "Red"

        return (steers, accels)


    def collision_lookahead(self, steers, accels, truth=False):
        '''
        Function to detect a collision given a vector of
        steering and acceleration actions.

        Propagates self and neighbors forward using bicycle model.
        If any step in the propagation has a collision returns a
        nonzero value.

        Set truth to True to get whether the agent actually collided or not.
        And False to get it based on the agent's belief.
        '''

        # Get neighbors
        # neighbors = self.model.space.get_neighbors(self.pos, GET_NEIGHBOR_DIST, False)
        neighbors = self.get_neighbors()


        # print("# neighbors: {}".format(len(neighbors)))
        # If no neighbors, return 0
        if len(neighbors) == 0:
            return 0

        # Propagate self forward using action vectors
        new_pos_list = self.bicycle_lookahead(steers, accels, self.accuracy)

        # Assume neighbors take no actions
        no_actions = np.zeros(len(steers))

        # Check all neighbors
        for neighbor in neighbors:

            # Propagate neighbor forward
            neighbor_pos_list = neighbor.bicycle_lookahead( \
                no_actions, no_actions, self.accuracy)

            if truth:
                collision_status = self.collision(new_pos_list, neighbor, neighbor_pos_list, 0)
            else:
                collision_status = self.collision(new_pos_list, neighbor, neighbor_pos_list, self.safety_margin)

            # Check each step in propagation for a collision
            if collision_status > 0:
                return collision_status

        return 0

    def get_neighbors(self):
        neighbors = []
        for car in self.model.cars:
            if car.unique_id != self.unique_id:
                neighbors.append(car)

        return neighbors

    def bicycle_lookahead(self, steers, accels, accuracy):
        '''
        Propagate state using bicycle model and action vectors
        steers and accels.

        Returns an array of positions
        '''

        # Initialize with current speed, heading, position
        speed = self.speed
        heading = self.heading
        pos = self.pos

        # Initialize empty list to hold propagated position vector
        future_pos = []

        # Loop through action vectors and apply bicyle model
        for steer, accel in np.c_[steers, accels]:

            # Propagate forward
            (next_speed, next_heading, next_pos) = self.bicycle_model_acc( \
                steer, accel, speed, heading, pos, accuracy)

            # Keep track of position
            future_pos.append(next_pos)

            # Go to next step
            speed = next_speed
            heading = next_heading
            pos = next_pos

        return np.array(future_pos)


    def collision(self, new_pos_list, neighbor, neighbor_pos_list, safety_margin):
        '''
        Check for a collision. If there is one indicate
        whether self is the front or back car in the collision
        '''


        v_safe_space = (self.car_length + neighbor.car_length) * \
                        (1 + safety_margin) / 2
        h_safe_space = (self.car_width + neighbor.car_width) * \
                        (1 + safety_margin) / 2

        # Check each step in the position list
        for new_pos, neighbor_pos in zip(new_pos_list, neighbor_pos_list):
            if self.collision_overlap(new_pos[0], \
                                      neighbor_pos[0], \
                                      h_safe_space) \
               and \
               self.collision_overlap(new_pos[1], \
                                      neighbor_pos[1], \
                                      v_safe_space):
               if new_pos[1] > neighbor_pos[1]:
                    return COLLIDE_BACK
               else:
                    return COLLIDE_FRONT

        return 0

    def collision_overlap(self, x1, x2, margin):
        return abs(x2 - x1)  - margin < 0

    def maintain_speed(self):
        return 0

    def accelerate(self):
        return self.accel_mag

    def brake(self):
        return -self.accel_mag

    def go_straight(self):
        return 0

    def turn_left(self):
        return self.steer_mag

    def turn_right(self):
        return -self.steer_mag

    def bicycle_model_acc(self, steer, accel, speed, heading, pos, accuracy):
        '''
        Returns a tuple of (next_speed, next_heading, next_position) after running
        a single update step to the bicycle model.

        The controls for the bicycle model are self.steer and self.accel which
        control the steering angle in radians and the longitudinal acceleration.

        Reference: http://www.me.berkeley.edu/~frborrel/pdfpub/IV_KinematicMPC_jason.pdf
        '''

        # Sideslip
        # Beta = arctan(lr*tan(d)/(lr+lf))
        sideslip = np.arctan(self.lr*np.tan(steer)/(self.lr + self.lf))
        sideslip = wrap_angle(sideslip)

        # Speed
        next_speed = speed + accel
        if next_speed < MIN_SPEED:
            next_speed = MIN_SPEED

        # Heading
        # delta_heading = v*sin(B)/lr
        next_heading = heading + next_speed*np.sin(sideslip)/self.lr
        next_heading = wrap_angle(next_heading)

        # Position
        delta_x = next_speed*np.cos(next_heading + sideslip)
        delta_y = next_speed*np.sin(next_heading + sideslip)
        if random.random() > 0.5:
            delta_x += delta_x * (1 - accuracy)
        else:
            delta_x -= delta_x * (1 - accuracy)
        if random.random() > 0.5:
            delta_y += delta_y * (1 - accuracy)
        else:
            delta_y -= delta_y * (1 - accuracy)

        next_pos = pos + np.array((delta_x, delta_y))

        # Enforce boundary constraints
        next_pos = self.boundary_adj(next_pos)

        # print("car id: {}, sideslip: {:4.2f}, speed: {:4.2f}, heading: {:4.2f}, dx: {:4.2f}, dy: {:4.2f}, old pos x: {:4.2f}, old pos y: {:4.2f}, new pos x: {:4.2f}, new pos y: {:4.2f}".format(
             # self.unique_id, np.degrees(sideslip), self.speed, np.degrees(self.heading), delta_x, delta_y, self.pos[0], self.pos[1], next_pos[0], next_pos[1]))

        return (next_speed, next_heading, next_pos)


    def bicycle_model(self, steer, accel, speed, heading, pos):
        return self.bicycle_model_acc(steer, accel, speed, heading, pos, 1)


    def boundary_adj(self, pos):
        '''
        Do not allow position to exceed boundaries
        '''

        space = self.model.space

        xmin = space.x_max/2 - self.road_width/2
        xmax = space.x_max/2 + self.road_width/2

        x = min(max(pos[0], xmin), xmax - 1e-8)
        y = space.y_min + ((pos[1] - space.y_min) % space.height)
        if isinstance(pos, tuple):
            return (x, y)
        else:
            return np.array((x, y))

    def vel_components(self):
        return self.speed * np.array((np.cos(self.heading), np.sin(self.heading)))
