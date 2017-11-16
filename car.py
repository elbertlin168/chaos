import numpy as np
import random
from mesa import Agent

# Minimum distance from neighbors
RISK_TOLERANCE = 5
ATTENTION = 0.95

# Desired speed
TARGET_SPEED = 10

# Allowed error on speed.
# Warning: If the margin is too small relative to the
# accel magnitude it will be unstable
SPEED_MARGIN = 0.1

# How much accel/brake in each action
ACCEL_MAG = 0.3

# Desired heading
TARGET_HEADING = -np.radians(90)

# Allowed error on heading.
# Warning: If the margin is too small relative to the
# steer magnitude, it will be unstable
HEADING_MARGIN = np.radians(1)

# How much steering in each action
STEER_MAG = np.radians(0.5)


def wrap_angle(angles):
   '''
   Returns angle betwee -pi and pi
   '''
   return ( angles + np.pi) % (2 * np.pi ) - np.pi

class Car(Agent):
    '''
    '''

    def __init__(self, unique_id, model, pos, speed, heading,
                 risk_tolerance=RISK_TOLERANCE, attention=ATTENTION,
                 target_speed=TARGET_SPEED, speed_margin=SPEED_MARGIN,
                 accel_mag=ACCEL_MAG, heading_margin=HEADING_MARGIN,
                 steer_mag=STEER_MAG):
        '''
        '''
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.speed = speed
        self.heading = heading
        self.risk_tolerance = risk_tolerance
        self.attention = attention
        self.target_speed = target_speed
        self.speed_margin = speed_margin
        self.accel_mag = accel_mag
        self.heading_margin = heading_margin
        self.steer_mag = steer_mag

        # Initialize the bicycle model
        # These constants can be varied if we want to change
        # how the steering angle effects the turning kinematics
        self.lr = 1
        self.lf = 1

        # Initialize inputs to 0
        self.accel = 0 # longitudinal acceleration effects speed directly
        # steering angle in radians controls the angular velocity
        # via the bicycle model kinematics
        self.steer = 0

    def step(self):
        '''
        '''
        # Speed control
        speed_error = self.speed - self.target_speed
        if speed_error < -self.speed_margin:
            self.accelerate()
        elif speed_error > self.speed_margin:
            self.brake()
        else:
            self.maintain_speed()


        # Heading control
        heading_error = wrap_angle(self.heading - TARGET_HEADING)

        if heading_error < -self.heading_margin:
            self.turn_left()
        elif heading_error > self.heading_margin:
            self.turn_right()
        else:
            self.go_straight()

        # print('car id: {}, heading err: {:4.2f}, speed err: {:4.2f}, accel: {:4.2f}, steer: {:4.2f}'.format(
        #     self.unique_id, np.degrees(heading_error), speed_error, self.accel, np.degrees(self.steer)))

        # Run bicycle model
        (next_speed, next_heading, next_pos) = self.bicycle_model()

        # Enforce space boundaries
        next_pos = self.boundary_adj(next_pos)
        # Take action if no collision
        if self.collision(next_pos):
            speed_actions = [self.maintain_speed, self.accelerate, self.brake]
            heading_actions = [self.turn_left, self.turn_right, self.go_straight]
            for sa in speed_actions:
                collide = True
                for ha in heading_actions:
                    (next_speed, next_heading, next_pos) = self.bicycle_model()
                    next_pos = self.boundary_adj(next_pos)
                    if not self.collision(next_pos):
                        collide = False
                        break
                if not collide:
                    break
        self.model.space.move_agent(self, next_pos)
        self.speed = next_speed
        self.heading = next_heading
        # collision = self.collision(next_pos)
        # if not collision:
        #     self.model.space.move_agent(self, next_pos)
        #     self.speed = next_speed
        #     self.heading = next_heading
        # else:
        #     print('{} Collision!', self.unique_id)

        # print("car id: {:3}, speed: {:5.1f}, heading: {:5.1f}, pos x: {:5.1f}, pos y: {:5.1f}, steer: {:5.1f}, accel: {:5.1f}".format(
        #     self.unique_id, self.speed, np.degrees(self.heading),  self.pos[0], self.pos[1], np.degrees(self.steer), self.accel))

    def boundary_adj(self, pos):
        space = self.model.space
        x = min(max(pos[0], space.x_min), space.x_max - 1e-8)
        y = space.y_min + ((pos[1] - space.y_min) % space.height)
        if isinstance(pos, tuple):
            return (x, y)
        else:
            return np.array((x, y))

    def horizontally_aligned(self, x1, x2, car_width):
        right_car = max(x1, x2)
        left_car = min(x1, x2)
        rcar_lside = right_car - car_width * 0.5
        lcar_rside = left_car + car_width * 0.5
        return rcar_lside <= lcar_rside

    def vertically_aligned(self, y1, y2, car_length):
        front_car = max(y1, y2)
        back_car = min(y1, y2)
        fcar_bside = front_car - car_length * 0.5
        bcar_fside = back_car + car_length * 0.5
        return fcar_bside <= bcar_fside

    def collision(self, new_pos):
        risk_tolerance = 0.5
        accuracy = 0.95
        attention = 0.5
        car_width = 0.01
        car_length = 0.025
        v_safe_space = car_length * (1 - self.risk_tolerance)
        h_safe_space = car_width * (1 - self.risk_tolerance)
        neighbors = self.model.space.get_neighbors(self.pos, 5, False)
        colliding = False
        for neighbor in neighbors:
            _, _, neighbor_pos = neighbor.bicycle_model_acc(accuracy)
            x1 = new_pos[0]
            x2 = neighbor_pos[0]
            y1 = new_pos[1]
            y2 = neighbor_pos[1]
            if y2 > y1 and self.horizontally_aligned(x1, x2, car_width) and \
            (y2 - car_length * 0.5) - (y1 + car_length * 0.5) <= v_safe_space:
                colliding = True

            if self.vertically_aligned(y1, y2, car_length) and \
            ((x2 - car_width * 0.5) - (x1 + car_width * 0.5) <= h_safe_space \
            or (x1 - car_width * 0.5) - (x2 + car_width * 0.5) <= h_safe_space):
                colliding = True

        if random.random() > self.attention:
            colliding = not colliding
        return colliding

    def maintain_speed(self):
        self.accel = 0

    def accelerate(self):
        self.accel = self.accel_mag

    def brake(self):
        self.accel = -self.accel_mag

    def go_straight(self):
        self.steer = 0

    def turn_left(self):
        self.steer = self.steer_mag

    def turn_right(self):
        self.steer = -self.steer_mag

    def bicycle_model_acc(self, accuracy):
        '''
        Returns a tuple of (next_speed, next_heading, next_position) after running
        a single update step to the bicycle model.

        The controls for the bicycle model are self.steer and self.accel which
        control the steering angle in radians and the longitudinal acceleration.

        Reference: http://www.me.berkeley.edu/~frborrel/pdfpub/IV_KinematicMPC_jason.pdf
        '''

        # Sideslip
        # Beta = arctan(lr*tan(d)/(lr+lf))
        sideslip = np.arctan(self.lr*np.tan(self.steer)/(self.lr + self.lf))
        sideslip = wrap_angle(sideslip)

        # Speed
        next_speed = self.speed + self.accel

        # Heading
        # delta_heading = v*sin(B)/lr
        next_heading = self.heading + next_speed*np.sin(sideslip)/self.lr
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

        next_pos = self.pos + np.array((delta_x, delta_y))

        # print("car id: {}, sideslip: {:4.2f}, speed: {:4.2f}, heading: {:4.2f}, dx: {:4.2f}, dy: {:4.2f}, old pos x: {:4.2f}, old pos y: {:4.2f}, new pos x: {:4.2f}, new pos y: {:4.2f}".format(
        #      self.unique_id, np.degrees(self.sideslip), self.speed, np.degrees(self.heading), delta_x, delta_y, self.pos[0], self.pos[1], new_pos[0], new_pos[1]))

        return (next_speed, next_heading, next_pos)


    def bicycle_model(self):
        return self.bicycle_model_acc(1)
