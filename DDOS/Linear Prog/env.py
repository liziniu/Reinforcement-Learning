import numpy as np
np.random.seed(1)


class Env:
    def __init__(self, n_agents):
        self.n_agents = n_agents

    def reset(self):
        total_flow = np.random.uniform(0, 1, self.n_agents)    # shape: (108, )
        normal_flow = np.random.uniform(0, 1, self.n_agents)   # shape: (108, )
        info = [total_flow, normal_flow]
        return total_flow, info

    def step(self, action):
        """
        这里是处理action的代码
        """
        # 处理完action后要更新t-2, t-1时刻的剩余流量
        # 所以total_flow, normal_flow是针对下一个时刻状态计算的
        total_flow = np.random.uniform(0, 1, self.n_agents)    # shape: (54, )
        normal_flow = np.random.uniform(0, 1, self.n_agents)   # shape: (54, )
        reward = np.random.uniform(0, 1)
        info = [total_flow, normal_flow]
        return total_flow, reward, info