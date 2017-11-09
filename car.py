import numpy as np

from mesa import Agent


class Car(Agent):
    '''
    '''

    def __init__(self, unique_id, model, pos, speed):
        '''
        '''
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.speed = speed

    def step(self):
        '''
        '''
        new_pos = self.pos + np.array((0, self.speed))
        collision = self.collision(new_pos)
        if not collision:
            self.model.space.move_agent(self, new_pos)

    def collision(self, new_pos):
        neighbors = self.model.space.get_neighbors(self.pos, 5, False)
        for neighbor in neighbors:
            if abs(new_pos[0] - neighbor.pos[0]) < 2:
                if abs(new_pos[1] - neighbor.pos[1]) < 2: 
                    return True
        return False
