import numpy as np

class RandomAgent:
    def __init__(self, agent_id, num_actions):
        self.agent_id = agent_id
        self.num_actions = num_actions

    def choose_action(self, observation):
        return np.random.randint(0, self.num_actions)
