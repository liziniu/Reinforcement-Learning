from env import Enviorment
from RL_brain import RL, CentralizedRL
import matplotlib.pyplot as plt

N_A = 2
MAX_EPISODES = 30

env = Enviorment()
agent_1, agent_2 = RL(N_A), RL(N_A)
agent = CentralizedRL(n_a=2, n_agents=2)
running_reward = 0
rewards = []
for i in range(MAX_EPISODES):
    a1, a2 = agent_1.choose_action(), agent_2.choose_action()
    joint_action = [a1, a2]
    reward = env.step(joint_action)
    agent_1.learn(a1, reward)
    agent_2.learn(a2, reward)

    rewards.append(reward/10)


for i in range(MAX_EPISODES):
    joint_action = agent.choose_action()
    reward = env.step(joint_action)
    agent.learn(joint_action, reward)
    rewards.append(reward/10)

print(agent_1.q)
print(agent_2.q)
print("*"*50)
print(agent.q)
plt.plot(rewards[:MAX_EPISODES])
plt.plot(rewards[MAX_EPISODES:])
plt.legend(["dec", "cen"])
plt.show()