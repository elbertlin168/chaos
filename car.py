import numpy as np
import random
from mesa import Agent
import math

# Minimum distance from neighbors
RISK_TOLERANCE = 0.5
ATTENTION = 0.95

# Desired speed
TARGET_SPEED = 3

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
STEER_MAG = np.radians(8)

# Lookahead
LOOKAHEAD = 5


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

    def choose_action(self):
        '''
        '''
        # Speed control
        speed_error = self.speed - self.target_speed
        if speed_error < -self.speed_margin:
            accel = self.accelerate()
        elif speed_error > self.speed_margin:
            accel = self.brake()
        else:
            accel = self.maintain_speed()


        # Heading control
        heading_error = wrap_angle(self.heading - TARGET_HEADING)

        if heading_error < -self.heading_margin:
            steer = self.turn_left()
        elif heading_error > self.heading_margin:
            steer = self.turn_right()
        else:
            steer = self.go_straight()

        # print('car id: {}, heading err: {:4.2f}, speed err: {:4.2f}, accel: {:4.2f}, steer: {:4.2f}'.format(
        #     self.unique_id, np.degrees(heading_error), speed_error, self.accel, np.degrees(self.steer)))

        # Run bicycle model
        steers = np.zeros(LOOKAHEAD)
        accels = np.zeros(LOOKAHEAD)
        steers[0] = steer
        accels[0] = accel

        # Enforce space boundaries
        # next_pos = self.boundary_adj(next_pos)
        # Take action if no collision
        # if self.unique_id > 0:
        #     print("car id: {:3}, cact. speed: {:5.1f}, heading: {:5.1f}, pos x: {:5.1f}, pos y: {:5.1f}, steer: {:5.1f}, accel: {:5.1f}".format(
        #          self.unique_id, self.speed, np.degrees(self.heading),  self.pos[0], self.pos[1], np.degrees(self.steer), self.accel))

        # if self.unique_id > 0:
            # print("first collision check")
        if self.collision_look_ahead(steers, accels):
            # if self.unique_id > 0:
            #     print('collision on first check')
            speed_actions = [self.maintain_speed, self.accelerate, self.brake]
            heading_actions = [self.turn_left, self.turn_right, self.go_straight]
            for sa in speed_actions:
                collide = True
                for ha in heading_actions:
                    accel = sa()
                    steer = ha()
                    steers.fill(steer)
                    accels.fill(accel)
                    # if self.unique_id > 0:
                    #     print("try: steer {:5.1f} accel{:5.1f}".format(np.degrees(steer), accel))
                    # (next_speed, next_heading, next_pos) = self.bicycle_model()
                    # next_pos = self.boundary_adj(next_pos)
                    if not self.collision_look_ahead(steers, accels):
                        # if self.unique_id > 0:
                        #     print("action chosen: steer {:5.1f} accel{:5.1f}".format(np.degrees(steer), accel))
                        collide = False
                        break
                if not collide:
                    break

        # if self.unique_id > 0:
        #     print("action at end of choose action: steer {:5.1f} accel{:5.1f}".format(np.degrees(steer), accel))

        self.steer = steer
        self.accel = accel
        # When this loop completes the steer and accel fields will have
        # been set based on the chosen action

        # self.speed = next_speed
        # self.heading = next_heading

        # collision = self.collision(next_pos)
        # if not collision:
        #     self.model.space.move_agent(self, next_pos)
        #     self.speed = next_speed
        #     self.heading = next_heading
        # else:
        #     print('{} Collision!', self.unique_id)


    def step(self):
        # if self.unique_id > 0:
        #     print("action taken: steer {:5.1f} accel{:5.1f}".format(np.degrees(self.steer), self.accel))

        (next_speed, next_heading, next_pos) = self.bicycle_model(self.steer, self.accel, self.speed, self.heading, self.pos)

        self.speed = next_speed
        self.heading = next_heading
        self.model.space.move_agent(self, next_pos)

        # if self.unique_id > 0:
        #     print("car id: {:3}, step. speed: {:5.1f}, heading: {:5.1f}, pos x: {:5.1f}, pos y: {:5.1f}, steer: {:5.1f}, accel: {:5.1f}".format(
        #          self.unique_id, self.speed, np.degrees(self.heading),  self.pos[0], self.pos[1], np.degrees(self.steer), self.accel))


    def boundary_adj(self, pos):
        space = self.model.space
        x = min(max(pos[0], space.x_min), space.x_max - 1e-8)
        y = space.y_min + ((pos[1] - space.y_min) % space.height)
        if isinstance(pos, tuple):
            return (x, y)
        else:
            return np.array((x, y))


    def collision_look_ahead(self, steers, accels):
        accuracy = 1
        new_pos_list = self.bicycle_lookahead(steers, accels, 1)

        no_actions = np.zeros(len(steers))

        neighbors = self.model.space.get_neighbors(self.pos, 500, False)
        # if self.unique_id > 0:
            # print("N neighbors:{}".format(len(neighbors)))
        colliding = False
        for neighbor in neighbors:
            neighbor_pos_list = neighbor.bicycle_lookahead(no_actions, no_actions, accuracy)
            for new_pos, neighbor_pos in zip(new_pos_list, neighbor_pos_list):
                if self.collision(new_pos, neighbor_pos):
                    return True

        return False


    def bicycle_lookahead(self, steers, accels, accuracy):
        speed = self.speed
        heading = self.heading
        pos = self.pos
        future_pos = []
        for steer, accel in np.c_[steers, accels]:
            (next_speed, next_heading, next_pos) = self.bicycle_model_acc(steer, accel, speed, heading, pos, accuracy)

            future_pos.append(next_pos)
            speed = next_speed
            heading = next_heading
            pos = next_pos

        return np.array(future_pos)
     

    def collision_overlap(self, x1, x2, margin):
        
        distance = abs(x2 - x1)  - margin

        # if self.unique_id > 0:
        #     print("x1: {:5.1f} x2: {:5.1f} margin: {:5.1f}, distance {:5.1f}".format(
        #         x1, x2, margin, distance))

        return distance < 0

    # def vertically_aligned(self, y1, y2, car_length):
    #     front_car = max(y1, y2)
    #     back_car = min(y1, y2)
    #     fcar_bside = front_car - car_length * 0.5
    #     bcar_fside = back_car + car_length * 0.5

    #     if self.unique_id > 0:
    #         print("x1: {:5.1f} y1: {:5.1f} x2: {:5.1f} y2: {:5.1f}, distance {:5.1f}".format(
    #             x1, y1, x2, y2, distance))

    #     return fcar_bside <= bcar_fside

    # def hcheck():
    #             if y2 < y1 and self.horizontally_aligned(x1, x2, car_width) and \
    #     (y1 - car_length * 0.5) - (y2 + car_length * 0.5) <= v_safe_space:
    #         colliding = True

    # def vcheck():
    #             if self.vertically_aligned(y1, y2, car_length) and \
    #     ((x2 - car_width * 0.5) - (x1 + car_width * 0.5) <= h_safe_space \
    #     or (x1 - car_width * 0.5) - (x2 + car_width * 0.5) <= h_safe_space):

    def collision(self, new_pos, neighbor_pos):
        safety_margin = .2
        accuracy = 1
        attention = 1
        car_width = 5 #12
        car_length = 25 #100
        v_safe_space = car_length * (1 + safety_margin)
        h_safe_space = car_width * (1 + safety_margin)

        x1 = new_pos[0]
        x2 = neighbor_pos[0]
        y1 = new_pos[1]
        y2 = neighbor_pos[1]

        colliding = False 
        if self.collision_overlap(x1, x2, h_safe_space) and self.collision_overlap(y1, y2, v_safe_space):
            colliding = True




        # if self.unique_id > 0:
        #     print("x1: {:5.1f} y1: {:5.1f} x2: {:5.1f} y2: {:5.1f}, haligned: {}, valigned: {}, dy {:5.1f}, safespace {:5.1f}".format(
        #         x1, y1, x2, y2, self.horizontally_aligned(x1, x2, car_width), 
        #         self.vertically_aligned(y1, y2, car_length), (y1 - car_length * 0.5) - (y2 + car_length * 0.5), 
        #         v_safe_space))

        # colliding = False
        # if y2 < y1 and self.horizontally_aligned(x1, x2, car_width) and \
        # (y1 - car_length * 0.5) - (y2 + car_length * 0.5) <= v_safe_space:
        #     colliding = True

        # if self.vertically_aligned(y1, y2, car_length) and \
        # ((x2 - car_width * 0.5) - (x1 + car_width * 0.5) <= h_safe_space \
        # or (x1 - car_width * 0.5) - (x2 + car_width * 0.5) <= h_safe_space):
        #     colliding = True

        # if self.unique_id > 0:
        #     print("collision? {}".format(colliding) )

        # if random.random() > self.attention:
            # colliding = not colliding

        # if self.unique_id > 0:
            # print("collision summary? {}".format(colliding) )
        return colliding

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

        next_pos = self.boundary_adj(next_pos)

        # print("car id: {}, sideslip: {:4.2f}, speed: {:4.2f}, heading: {:4.2f}, dx: {:4.2f}, dy: {:4.2f}, old pos x: {:4.2f}, old pos y: {:4.2f}, new pos x: {:4.2f}, new pos y: {:4.2f}".format(
             # self.unique_id, np.degrees(sideslip), self.speed, np.degrees(self.heading), delta_x, delta_y, self.pos[0], self.pos[1], next_pos[0], next_pos[1]))

        return (next_speed, next_heading, next_pos)


    def bicycle_model(self, steer, accel, speed, heading, pos):
        return self.bicycle_model_acc(steer, accel, speed, heading, pos, 1)
