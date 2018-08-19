import random
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, delay):
        pass
        self.current = random.uniform(1, 10)
        self.others = 0
        self.delay = delay
        self.time = 1

    def perceive(self):
        self.current = random.randint(1, 10)

    def send(self, table):
        info = [self.delay+self.time, self.current]
        table.append(info)
        self.time += 1

    def receive(self, others):
        self.others = others[1]

    def get_total(self):
        return self.others + self.current


MAX_TIME = 100
agent_a = Agent(delay=4)
agent_b = Agent(delay=4)
table_a2b = []
table_b2a = []

ts = []
tas = []
tbs = []
das = []
dbs = []
for i in range(1, MAX_TIME+1):
    agent_a.perceive()
    agent_a.send(table_a2b)
    agent_b.perceive()
    agent_b.send(table_b2a)

    for t in table_a2b:
        if t[0] != i:
            break
        else:
            agent_b.receive(t)
            table_a2b.remove(t)

    for t in table_b2a:
        if t[0] != i:
            break
        else:
            agent_a.receive(t)
            table_b2a.remove(t)

    ts.append(agent_a.current + agent_b.current)
    tas.append(agent_a.get_total())
    tbs.append(agent_b.get_total())
    das.append(agent_a.current + agent_b.current - agent_a.get_total())
    dbs.append(agent_a.current + agent_b.current - agent_b.get_total())
    if i <= 5:
        print('t:', i, ',a_current: ', agent_a.current, ',b_current:', agent_b.current,
              ',total_a:', agent_a.get_total(), ',total_b:', agent_b.get_total())

plt.style.use('seaborn')
plt.figure()
plt.plot(ts, )
plt.plot(tas, )
plt.plot(tbs, )
plt.legend(['total', 'total_a', 'total_b'])

plt.figure()
plt.plot(das)
plt.plot(dbs)
plt.legend(['diff_a', 'diff_b'])
plt.show()







