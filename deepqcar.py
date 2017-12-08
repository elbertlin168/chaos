import random
import numpy as np
from car import Car

from scipy.ndimage.interpolation import rotate

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam, RMSprop
from keras import backend as K

from util import get_bin
from settings import Actions

STATE_SHAPE = (4, 100, 100)
WEIGHT_FILE = "maybe-chaos-ddqn.h5"

HUBER_LOSS_DELTA = 2

# Field of view, Manhattan distance
FOV = 25

def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2, L1)   # Keras does not cover where function in tensorflow :-(

    return K.mean(loss)

class DeepQCar(Car):

    def __init__(self, *args, epsilon=1.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.state_shape = STATE_SHAPE
        self.current_state = np.zeros(self.state_shape)
        self.action = random.choice(list(Actions))

        # self.memory = deque(maxlen=2000)
        self.gamma = 0.99         # discount rate
        self.epsilon = epsilon    # exploration rate
        # self.epsilon_min = 0.01
        # self.epsilon_decay = 0.995
        self.learning_rate = 0.00025
        self.nn = self._build_nn()
        self.target_nn = self._build_nn()
        self.update_target_nn()

    def _build_nn(self):
        # Neural Net for Deep-Q learning Model
        nn = Sequential()
        nn.add(Conv2D(32, (8,8), strides=(4,4), activation='relu',
                      input_shape=self.state_shape,
                      data_format='channels_first'))
        nn.add(Conv2D(64, (4,4), strides=(2,2), activation='relu'))
        nn.add(Conv2D(64, (3,3), activation='relu'))
        nn.add(Flatten())
        nn.add(Dense(512, activation='relu'))
        nn.add(Dense(len(Actions), activation='linear'))
        opt = RMSprop(lr=self.learning_rate)
        nn.compile(loss=huber_loss, optimizer=opt)
        try:
            open(WEIGHT_FILE, "r")
            nn.load_weights(WEIGHT_FILE)
        except:
            pass
        return nn

    def update_target_nn(self):
        self.target_nn.set_weights(self.nn.get_weights())

    def save(self, name=WEIGHT_FILE):
        self.nn.save_weights(name)

    # def remember(self, state, action, reward, next_state, running):
    #     self.memory.append((state, action, reward, next_state, running))

    def replay(self, memory, batch_size):
        minibatch = np.array(random.sample(memory, batch_size))
        for state, action, reward, next_state, running in minibatch:
            state = state.reshape(1, 4, 100, 100)
            next_state = next_state.reshape(1, 4, 100, 100)
            target = self.nn.predict(state)
            if running:
                a = self.nn.predict(next_state)[0]
                t = self.target_nn.predict(next_state)[0]
                target[0][action.value - 1] = reward + self.gamma * \
                        t[np.argmax(a)]
            else:
                target[0][action.value - 1] = reward
            self.nn.fit(state, target, epochs=1, verbose=0)
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

    def choose_action(self):
        # if self.model.count < 0:
        #     super().choose_action()
        #     if self.steer == self.go_straight():
        #         act_val = 1
        #     elif self.steer == self.turn_left():
        #         act_val = 4
        #     else:
        #         act_val = 7
        #     if self.accel == self.accelerate():
        #         act_val += 1
        #     elif self.accel == self.brake():
        #         act_val += 2
        #     self.action = Actions(act_val)
        #     return
        if random.random() <= self.epsilon:
            self.action = random.choice(list(Actions))
        else:
            act_rewards = self.nn.predict(self.current_state.reshape(1, 4, 100, 100))
            self.action = Actions(np.argmax(act_rewards[0]) + 1)

        # if self.action.value % 3 == 1:           # Actions.{X}_M
        #     self.accel = self.maintain_speed()
        # elif self.action.value % 3 == 2:         # Actions.{X}_A
        #     self.accel = self.accelerate()
        # else:                                    # Actions.{X}_B
        #     self.accel = self.brake()

        if self.action.value == 1:    # Actions.S_{X}
            self.steer = self.go_straight()
        elif self.action.value == 2:  # Actions.L_{X}
            self.steer = self.turn_left()
        else:                                    # Actions.R_{X}
            self.steer = self.turn_right()

    def step(self):
        self.reward = self.model.deepq_reward(self)
        self.update_state_grid()
        (self.speed, self.heading, next_pos) = self.bicycle_model( \
            self.steer, self.accel, self.speed, self.heading, self.pos)
        self.heading = -np.radians(90)
        self.model.space.move_agent(self, next_pos)

    def current_grid(self):
        # Visualized as if we placed a discrete grid on top of model space,
        # centered on our agent. m is row (y value), n is column (x value)
        m, n = self.state_shape[1:3]
        grid = np.zeros((m, n))
        x_min = self.pos[0] - FOV
        x_bin_size = FOV * 2 / n
        y_min = self.pos[1] - FOV
        y_bin_size = FOV * 2 / m

        # All cars within FOV (automatically pruned with range)
        for car in self.model.cars:
            val = 0.5 if car.unique_id == self.unique_id else 1
            car_top = car.pos[1] - car.length / 2
            car_bottom = car_top + car.length
            i_min = max(0, get_bin(car_top, y_min, y_bin_size))
            i_max = min(m, get_bin(car_bottom, y_min, y_bin_size) + 1)

            car_left = car.pos[0] - car.width / 2
            car_right = car_left + car.width
            j_min = max(0, get_bin(car_left, x_min, x_bin_size))
            j_max = min(n, get_bin(car_right, x_min, x_bin_size) + 1)

            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    grid[i, j] = val

        # Road edges
        left = (self.model.space.x_max  - self.model.road_width) / 2
        j_max = min(100, get_bin(left, x_min, x_bin_size))
        right = left + self.model.road_width
        j_min = max(0, get_bin(right, x_min, x_bin_size))
        for i in range(m):
            for j in range(j_max):
                grid[i, j] = 2
            for j in range(j_min, n):
                grid[i, j] = 2

        # return rotate(grid, -np.degrees(self.heading), reshape=False, mode="nearest")
        return grid

    def update_state_grid(self):
        self.current_state = np.array([self.current_state[1],
                                       self.current_state[2],
                                       self.current_state[3],
                                       self.current_grid()])
        # self.current_state[:,:,:,0] = self.current_grid()

    def boundary_adj(self, pos):
        '''
        Do not allow position to exceed boundaries
        '''

        space = self.model.space

        x = pos[0]
        y = space.y_min + ((pos[1] - space.y_min) % space.height)
        if isinstance(pos, tuple):
            return (x, y)
        else:
            return np.array((x, y))
