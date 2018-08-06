import numpy as np


class RL:
    def __init__(self, n_a, gamma=0.99, epsilon=0.5, learning_rate=0.05):
        self.q = np.zeros(n_a, dtype=np.float32)
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.increment = 0.03
        pass

    def learn(self, action, reward):
        self.q[action] += self.learning_rate * (reward - self.q[action])
        self.epsilon += self.increment if self.epsilon < 0.95 else 0
        pass

    def choose_action(self):
        if np.random.uniform() < self.epsilon:
            return np.argmax(self.q)
        else:
            np.random.choice([0, 1])


class CentralizedRL:
    def __init__(self, n_agents, n_a, epsilon=0.5, learning_rate=0.05):
        self.q = np.zeros(n_agents*n_a)
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.increment = 0.03
        pass

    def learn(self, action, reward):
        if action == [0, 0]:
            action = 0
        elif action == [0, 1]:
            action = 1
        elif action == [1, 0]:
            action = 2
        else:
            action = 3
        self.q[action] += self.learning_rate * (reward - self.q[action])
        self.epsilon += self.increment if self.epsilon < 0.95 else 0

        pass

    def choose_action(self):
        if np.random.uniform() < self.epsilon:
            index = np.argmax(self.q)
            if index == 0:
                return [0, 0]
            elif index == 1:
                return [0, 1]
            elif index == 2:
                return [1, 0]
            else:
                return [1, 1]
        else:
            return [np.random.choice([0, 1]), np.random.choice([0, 1])]

if __name__ == "__main__":
    pass
