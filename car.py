import numpy as np

from mesa import Agent

# Minimum distance from neighbors
SAFE_DIST = 5

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
STEER_MAG = np.radians(.5)


def wrap_angle(angles):
   '''
   Returns angle betwee -pi and pi
   '''
   return ( angles + np.pi) % (2 * np.pi ) - np.pi

class Car(Agent):
    '''
    '''

    def __init__(self, unique_id, model, pos, speed, heading,
                 safe_dist=SAFE_DIST, target_speed=TARGET_SPEED,
                 speed_margin=SPEED_MARGIN, accel_mag=ACCEL_MAG,
                 heading_margin=HEADING_MARGIN, steer_mag=STEER_MAG):
        '''
        '''
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.speed = speed
        self.heading = heading
        self.safe_dist = safe_dist
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
        speed_error = self.speed - TARGET_SPEED
        if speed_error < -SPEED_MARGIN:
            self.accelerate()
        elif speed_error > SPEED_MARGIN:
            self.brake()
        else:
            self.maintain_speed()


        # Heading control
        heading_error = wrap_angle(self.heading - TARGET_HEADING)

        if heading_error < -HEADING_MARGIN:
            self.turn_left()
        elif heading_error > HEADING_MARGIN:
            self.turn_right()
        else:
            self.go_straight()

        # print('car id: {}, heading err: {:4.2f}, speed err: {:4.2f}, accel: {:4.2f}, steer: {:4.2f}'.format(
        #     self.unique_id, np.degrees(heading_error), speed_error, self.accel, np.degrees(self.steer)))

        # Run bicycle model
        (next_speed, next_heading, next_pos) = self.bicycle_model()

        # Enforce space boundaries
        space = self.model.space
        next_pos[0] = min(max(next_pos[0], space.x_min), space.x_max - 1e-8)
        next_pos[1] = space.y_min + ((next_pos[1] - space.y_min) % space.height)
        # Take action if no collision
        collision = self.collision(next_pos)
        if not collision:
            space.move_agent(self, next_pos)
            self.speed = next_speed
            self.heading = next_heading
        else: 
            print('{} Collision!', self.unique_id)

        print("car id: {:3}, speed: {:5.1f}, heading: {:5.1f}, pos x: {:5.1f}, pos y: {:5.1f}, steer: {:5.1f}, accel: {:5.1f}".format(
            self.unique_id, self.speed, np.degrees(self.heading),  self.pos[0], self.pos[1], np.degrees(self.steer), self.accel))
        

    def collision(self, new_pos):
        neighbors = self.model.space.get_neighbors(self.pos, 5, False)
        for neighbor in neighbors:
            if abs(new_pos[0] - neighbor.pos[0]) < 2:
                if abs(new_pos[1] - neighbor.pos[1]) < 2: 
                    return True
        return False

    def maintain_speed(self):
        self.accel = 0

    def accelerate(self):
        self.accel = ACCEL_MAG

    def brake(self):
        self.accel = -ACCEL_MAG

    def go_straight(self):
        self.steer = 0

    def turn_left(self):
        self.steer = STEER_MAG

    def turn_right(self):
        self.steer = -STEER_MAG

    def bicycle_model(self):
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

        next_pos = self.pos + np.array((delta_x, delta_y))

        # print("car id: {}, sideslip: {:4.2f}, speed: {:4.2f}, heading: {:4.2f}, dx: {:4.2f}, dy: {:4.2f}, old pos x: {:4.2f}, old pos y: {:4.2f}, new pos x: {:4.2f}, new pos y: {:4.2f}".format(
        #      self.unique_id, np.degrees(self.sideslip), self.speed, np.degrees(self.heading), delta_x, delta_y, self.pos[0], self.pos[1], new_pos[0], new_pos[1]))
        
        return (next_speed, next_heading, next_pos)

