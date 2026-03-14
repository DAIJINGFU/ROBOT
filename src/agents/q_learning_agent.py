import numpy as np
import pickle
import os

class QLearningAgent:
    def __init__(self, agent_id, action_space_size, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.agent_id = agent_id
        self.action_space_size = action_space_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = {} # 使用字典存储稀疏的 Q 值: state -> [q_val_0, q_val_1, ...]

    def get_q(self, state, action):
        return self.q_table.get(state, np.zeros(self.action_space_size))[action]

    def choose_action(self, state):
        """Epsilon-greedy 策略"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space_size)
        
        if state not in self.q_table:
            return np.random.randint(self.action_space_size)
        
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        """Q-learning 更新公式"""
        # 获取当前 Q 值
        current_q_values = self.q_table.get(state, np.zeros(self.action_space_size))
        current_q = current_q_values[action]
        
        # 获取下一状态的最大 Q 值
        next_q_values = self.q_table.get(next_state, np.zeros(self.action_space_size))
        max_next_q = np.max(next_q_values)
        
        # 更新公式
        # Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s', a') - Q(s,a)]
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        
        # 存回表里
        current_q_values[action] = new_q
        self.q_table[state] = current_q_values

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.q_table = pickle.load(f)
