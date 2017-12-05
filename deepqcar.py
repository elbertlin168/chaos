import random
import numpy as np
from car import Car

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam

from util import get_bin
from settings import Actions

STATE_SHAPE = (1, 100, 100, 4)
WEIGHT_FILE = "chaos-dqn.h5"

# Field of view, Manhattan distance
FOV = 25

class DeepQCar(Car):

    def __init__(self, *args, epsilon=1.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.state_shape = STATE_SHAPE
        self.current_state = np.zeros(self.state_shape)
        self.action = random.choice(list(Actions))

        # self.memory = deque(maxlen=2000)
        self.gamma = 0.95         # discount rate
        self.epsilon = epsilon    # exploration rate
        # self.epsilon_min = 0.01
        # self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.nn = self._build_nn()

    def _build_nn(self):
        # Neural Net for Deep-Q learning Model
        nn = Sequential()
        nn.add(Conv2D(16, 8, strides=4, activation='relu',
                      input_shape=self.state_shape[1:]))
        nn.add(Conv2D(32, 4, strides=2, activation='relu'))
        nn.add(Flatten())
        nn.add(Dense(256, activation='relu'))
        nn.add(Dense(len(Actions), activation='linear'))
        nn.compile(loss='mse',
                   optimizer=Adam(lr=self.learning_rate))
        try:
            open(WEIGHT_FILE, "r")
            nn.load_weights(WEIGHT_FILE)
        except:
            pass
        return nn

    def save(self, name=WEIGHT_FILE):
        self.nn.save_weights(name)

    # def remember(self, state, action, reward, next_state, running):
    #     self.memory.append((state, action, reward, next_state, running))

    def replay(self, memory, batch_size):
        minibatch = np.array(random.sample(memory, batch_size))
        for state, action, reward, next_state, running in minibatch:
            target = reward
            if running:
                target = reward + self.gamma * \
                         np.amax(self.nn.predict(next_state)[0])
            target_f = self.nn.predict(state)
            target_f[0][action.value - 1] = target
            self.nn.fit(state, target_f, epochs=1, verbose=0)
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

    def choose_action(self):
        if random.random() <= self.epsilon:
            self.action = random.choice(list(Actions))
        else:
            act_rewards = self.nn.predict(self.current_state)
            self.action = Actions(np.argmax(act_rewards[0]) + 1)

        if self.action.value % 3 == 1:           # Actions.{X}_M
            self.accel = self.maintain_speed()
        elif self.action.value % 3 == 2:         # Actions.{X}_A
            self.accel = self.accelerate()
        else:                                    # Actions.{X}_B
            self.accel = self.brake()

        if (self.action.value - 1) // 3 == 0:    # Actions.S_{X}
            self.steer = self.go_straight()
        elif (self.action.value - 1) // 3 == 1:  # Actions.L_{X}
            self.steer = self.turn_left()
        else:                                    # Actions.R_{X}
            self.steer = self.turn_right()

    def step(self):
        self.reward = self.model.reward(self)
        self.update_state_grid()
        super().step()

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
            car_top = car.pos[1] - car.length / 2
            car_bottom = car_top + car.length
            i_min = max(0, get_bin(car_top, y_min, y_bin_size))
            i_max = min(m, get_bin(car_bottom, y_min, y_bin_size))

            car_left = car.pos[0] - car.width / 2
            car_right = car_left + car.width
            j_min = max(0, get_bin(car_left, x_min, x_bin_size))
            j_max = min(n, get_bin(car_right, x_min, x_bin_size))

            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    grid[i, j] = 1

        # Road edges
        left = (self.model.space.x_max  - self.model.road_width) / 2
        right = left + self.model.road_width
        for i in range(m):
            for j in range(get_bin(left, x_min, x_bin_size)):
                grid[i, j] = 1
            for j in range(get_bin(right, x_min, x_bin_size), n):
                grid[i, j] = 1

        return grid

    def update_state_grid(self):
        self.current_state = self.current_state[:,:,:,[3,0,1,2]]
        self.current_state[:,:,:,0] = self.current_grid()
