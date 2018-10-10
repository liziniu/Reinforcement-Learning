import matplotlib.pyplot as plt
import tensorflow as tf
from ddpg import  DDPG
from env import Env
import numpy as np


N_AGENTS = 108
Gamma = [0.1]
MEMORY_SIZE = 20000
record_file = open("ddpg-test.txt", "w")
env = Env(N_AGENTS)

sess = tf.Session()
tf.set_random_seed(1)
plt.style.use('seaborn')

for gamma in Gamma:
    agents = DDPG(n_agents=N_AGENTS, gamma=gamma, memory_size=MEMORY_SIZE, train=True)

    s, info = env.reset()    # 注意这里!!!
    rs = []
    epi_r = 0
    for i in range(10000):
        a = agents.choose_action(s)

        s_, r, info_ = env.step(a)   # 注意这里!!! info is a list [total_flow, normal_flow]

        agents.store_transition(s, a, r, s_, info)   # 注意这里!!! info和s对应；info_和s_对应
        rs.append(r)
        epi_r = 0.1 * r + 0.9 * epi_r

        if i > 64 and i < 10000 -1000:   # 最后1000个时间步用做测试
            agents.learn()

        s = s_
        info = info_     # 注意这里!!!

        if i % 1000 == 0:
            print(i, ",", epi_r)
    test_reward = np.mean(np.array(rs)[-1000: ])
    record_file.writelines(str(test_reward) + "|")

    plt.figure()
    plt.plot(rs)
    plt.title(str(gamma))
    plt.savefig("ddpg" + str(gamma) + ".png")