import random
import numpy as np
from car import Car

from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D
from keras.optimizers import Adam

STATE_SIZE = 5
ACTION_SPACE = 9


class DeepQCar(Car):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.nn = self._build_nn()

        self.state_size = STATE_SIZE
        self.state_shape = (STATE_SIZE, STATE_SIZE, 4)
        self.current_state = np.zeros((STATE_SIZE, STATE_SIZE, 4))

    def _build_nn(self):
        # Neural Net for Deep-Q learning Model
        nn = Sequential()
        nn.add(Conv2D(16, 8, strides=4, activation='relu', input_shape=self.state_shape))
        nn.add(Conv2D(32, 4, strides=2, activation='relu'))
        nn.add(Dense(256, activation='relu'))
        n.add(Dense(ACTION_SPACE, activation='linear'))
        n.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def load(self, name):
        self.nn.load_weights(name)

    def save(self, name):
        self.nn.save_weights(name)

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            target = reward
            # if not self.model.is_terminal(next_state):
            #     target = (reward + self.gamma *
            #               np.amax(self.model.reward(next_state)))
            # target_f = self.model.predict(state)
            # target_f[0][action] = target
            # self.nn.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def choose_action(self):
        super().choose_action()

    def get_grid(self, new_neighbors_pos):
        new_pos = self.pos
        n = self.state_size
        grid = np.zeros((n, n))
        grid[int(n / 2)][int(n / 2)] = 100 # our agent

        space = self.model.space
        xmin = space.x_max/2 - self.model.road_width/2
        xmax = space.x_max/2 + self.model.road_width/2

        width_size = 50.0 / n
        height_size = 200.0 / n

        # set boundaries
        if abs(xmin - new_pos[0]) <= 25:
            boundary_min = 0
            if abs(xmin - new_pos[0]) < width_size / 2.0:
                boundary_min = int(n / 2)
            elif xmin > new_pos[0]:
                boundary_min = int((xmin - new_pos[0] - width_size / 2.0) / width_size) + int(n / 2) + 1
            else:
                boundary_min = int((xmin - new_pos[0] + width_size / 2.0) / width_size) + int(n / 2) - 1
            for i in range(n):
                grid[i][boundary_min] += -5

        if abs(xmax - new_pos[0]) <= 25:
            boundary_max = 0
            if abs(xmax - new_pos[0]) < width_size / 2.0:
                boundary_max = int(n / 2)
            elif xmax > new_pos[0]:
                boundary_max = int((xmax - new_pos[0] - width_size / 2.0) / width_size) + int(n / 2) + 1
            else:
                boundary_max = int((xmax - new_pos[0] + width_size / 2.0) / width_size) + int(n / 2) - 1
            for i in range(n):
                grid[i][boundary_max] += -10

        for neighbor in new_neighbors_pos:
            rel_x = neighbor[0] - new_pos[0]
            rel_y = neighbor[1] - new_pos[1]
            if abs(rel_y) <= 100 and abs(rel_x) <= 25:
                x = 0
                y = 0
                if abs(rel_x) < width_size / 2:
                    x = int(n / 2)
                elif rel_x > 0:
                    x = int((rel_x - width_size / 2) / width_size) + int(n / 2) + 1
                else:
                    x = int((rel_x + width_size / 2) / width_size) + int(n / 2) - 1
                if abs(rel_y) < height_size / 2:
                    y = int(n / 2)
                elif rel_y > 0:
                    y = int((rel_y - height_size / 2) / height_size) + int(n / 2) + 1
                else:
                    y = int((rel_y + height_size / 2) / height_size) + int(n / 2) - 1
                grid[y][x] += 1

    def update_state_grid(self):
        new_neighbors_pos = []
        for neighbor in self.get_neighbors():
            new_neighbors_pos.append(neighbor.pos)
        new_state = []
        current_grid = self.get_grid(new_neighbors_pos)

        self.current_state = self.current_state[:,:,[3,0,1,2]]
        self.current_state[:,:,0] = current_grid
