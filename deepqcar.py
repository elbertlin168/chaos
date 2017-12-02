import random
import numpy as np
import car

from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D
from keras.optimizers import Adam

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
