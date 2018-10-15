import tensorflow as tf
import numpy as np
import threading
from copy import copy

np.random.seed(1)
tf.set_random_seed(1)


class Mythread(threading.Thread):
    def __init__(self, func, args=()):
        super(Mythread, self).__init__()
        self.func = func
        self.args = args
        self.result = None

    def run(self):
        self.func(*self.args)


class DDPG:
    def __init__(self, n_agents, sess, memory_capacity=1e4, batch_size=128, gamma=0.05, train=True):
        """
        Communication with 1-bit
        :param n_agents: number of agents; int
        :param memory_capacity: memory size; int
        :param batch_size: sample batch size; int
        :param gamma: discount; float
        :param train: True(train)/False(test); boolean
        :param sess: tensorflow session
        """
        self.n_agents = n_agents
        self.s_dim = n_agents
        self.gamma = gamma

        # communication pool, np.array, shape:(n_agents, )
        # each entity is from -1 to 1.
        self.pool = np.clip(np.random.normal(loc=0, scale=0.3, size=n_agents), -1, 1)
        self.pool_copy = self.pool.copy()
        self.communicate_counter = 0
        self.train_counter = 0
        # transition:[state, pool, state_, pool_, , reward]
        self.memory_size = int(memory_capacity)
        self.batch_size = int(batch_size)
        self.memory_pointer = 0
        self.memory = np.zeros(shape=[self.memory_size, self.s_dim*2+self.n_agents*3+1], dtype=np.float32)

        self.coord = tf.train.Coordinator()
        self.sess = sess
        self.actors = []
        self.actions = []
        self.actions_ = []

        for i in range(n_agents):
            actor = Actor(name=i, sess=sess, s_dim=n_agents, var_decrement=memory_capacity, train=train)
            action = tf.concat([actor.a1, actor.a2], axis=1)
            action_ = tf.concat([actor.a1_, actor.a2_], axis=1)

            self.actors.append(actor)
            self.actions.append(action)
            self.actions_.append(action_)

        a4c = tf.concat(self.actions, axis=1)
        a4c_ = tf.concat(self.actions_, axis=1)
        self.critic = Critic(n_agents=n_agents, sess=sess, gamma=gamma, actions=a4c, actions_=a4c_)

        self.replace_target = [actor.soft_replace for actor in self.actors]
        self.replace_target.append(self.critic.soft_replace)

        self.r = tf.placeholder(tf.float32, [None, 1], "reward")

        self.critic_train, self.actor_train = self._get_trainer()

        self.saver = tf.train.Saver(max_to_keep=1)

        if train:
            self.sess.run(tf.global_variables_initializer())
        else:
            model_file = tf.train.latest_checkpoint('ckpt/')
            self.saver.restore(sess, model_file)

    def communicate(self):
        """
        Each agent perceive its own local flow and communicate info with each other. Finally each agent get its own obs.
        :param state: global flow info; np.array, shape: (n_agents,)
        :return: None
        """
        # update communication pool
        self.pool_copy = self.pool.copy()
        self.communicate_counter = 0
        for i in range(self.n_agents):
            actor = self.actors[i]
            communicate_action = actor.get_communicate_action()
            if communicate_action > 0:
                self.pool[i] = communicate_action
                self.communicate_counter += 1
        # if not (self.pool_copy == self.pool).all():
        #    print("True")

    def choose_action(self, state):
        """
        choose action about cut down flow
        :param state: global flow state; array, (s_dim, )
        :return: joint actions; array, (n_agents, )
        """
        # update obs
        actions = np.zeros(self.n_agents, dtype=np.float32)
        for i in range(self.n_agents):
            actor = self.actors[i]
            actor.update_obs(state, self.pool.copy())
            actions[i] = actor.get_cut_action()
        return actions

    def get_agent_communication_action(self, i):
        """
        Get agent's communication action; Only valid after agents.communicate() is called
        :param i: indices; int, or array with shape (n, )
        :return:  communication action; array, shape(1, ) or (n, )
        """
        return self.pool[i]

    def store_transition(self, s, a, r, s_):
        """
        Store experience
        :param s: global flow state; array, (n_agents, )
        :param s_: next global flow, state; array, (n_agents, )
        :param a: joint cut action; array, (n_agents, )
        :param r: joint reward; float
        :return: None
        """
        # Note sample include: s, pool, s_, pool_, action, r
        t = np.hstack([s, self.pool_copy, s_, self.pool.copy(), a, [r]])
        index = int(self.memory_pointer % self.memory_size)
        self.memory[index, :] = t
        self.memory_pointer += 1

    def learn(self):
        """
        Sample from memory and Train actors and critic
        :return: None
        """
        if self.train_counter % 1000 == 0:
            self.sess.run(self.replace_target)

        if self.memory_pointer < self.memory_size:
            indices = np.random.choice(self.memory_pointer, self.batch_size)
        else:
            indices = np.random.choice(self.memory_size, self.batch_size)
        batch = self.memory[indices, :]
        s = batch[:, :self.s_dim]
        pool = batch[:, self.s_dim: self.n_agents+self.s_dim]
        s_ = batch[:, self.n_agents+self.s_dim: self.n_agents+self.s_dim*2]
        pool_ = batch[:, self.n_agents+self.s_dim*2: self.n_agents*2 + self.s_dim*2]
        cut_a = batch[:, self.n_agents*2 + self.s_dim*2: self.n_agents*3 + self.s_dim*2]
        r = batch[:, -1]

        # reconstruct actions for critic from pool_ and cut_a
        a = np.concatenate([cut_a, pool_], axis=1)

        feed_dict_critic = {}
        feed_dict_critic[self.critic.s] = s
        feed_dict_critic[self.critic.s_] = s_
        feed_dict_critic[self.r] = r[:, np.newaxis]    # (32) - > (32, 1)
        feed_dict_critic[self.critic.a] = a

        feed_dict_actor = {}
        feed_dict_actor[self.critic.s] = s
        # 恢复obs
        for i in range(self.n_agents):
            obs = copy(pool)
            obs_ = copy(pool_)
            obs[:, i] = s[:, i]
            obs_[:, i] = s_[:, i]

            # the process of critic's train does not need actor eval network(obs),
            # but it need obs_ to get a_ for target q network
            feed_dict_critic[self.actors[i].input_] = obs_
            # update actor network does not need target network
            feed_dict_actor[self.actors[i].input] = obs

        # critic
        self.sess.run(self.critic_train, feed_dict=feed_dict_critic)
        # actor
        # threads = [Mythread(func=self.sess.run, args=(self.actor_train[i], feed_dict_actor)) for i in range(self.n_agents)]
        # work_threads = []
        # for t in threads:
        #     t.start()
        #     work_threads.append(t)
        # self.coord.join(work_threads)
        self.sess.run(self.actor_train, feed_dict=feed_dict_actor)
        self.train_counter += 1

    def save_model(self):
        """
        Save model parameters
        :return: None
        """
        self.saver.save(self.sess, 'ckpt-communication/ddpg.ckpt')

    def _get_trainer(self):
        q = self.critic.q
        q_target = self.r + self.gamma * self.critic.q_

        actor_train = []

        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)

        critic_train = self.critic.opt.minimize(td_error, var_list=self.critic.ce_para)

        actor_loss = - tf.reduce_mean(q)

        for i in range(self.n_agents):
            actor = self.actors[i]
            a_train = actor.opt.minimize(actor_loss, var_list=actor.ae_para)
            actor_train.append(a_train)

        return critic_train, actor_train


class Actor:
    def __init__(self, name, sess, s_dim, a_dim=1, tau=0.05, var_cut=0.9, var_com=0.6, var_decrement=1e-4, learning_rate=1e-3, train=True):
        """
        Single Actor
        :param name: identity; int
        :param sess: tensorflow session
        :param s_dim: state's dim; int
        :param a_dim: cut action's dim; int
        :param tau: ratio of replace target; float
        :param var_cut: variance of cut action; float
        :param var_com: variance of communication action; float
        :param learning_rate: learning rate of optimizer; float
        :param train: True(train)/False(test); boolean
        """
        self.a_dim = a_dim
        self.name = name
        self.sess = sess
        self.var_cut = var_cut
        self.var_com = var_com
        self.var_decrement = var_decrement
        self.train = train


        # with graph.as_default():
        with tf.variable_scope('Actor' + str(name)):
            self.input = tf.placeholder(tf.float32, [None, s_dim], 'obs')
            self.input_ = tf.placeholder(tf.float32, [None, s_dim], 'obs_')
            self.obs = np.zeros(s_dim, np.float32)

            self.a1, self.a2 = self._build_a(self.input, scope='eval', trainable=True)
            self.a1_, self.a2_ = self._build_a(self.input_, scope='target', trainable=False)

            self.ae_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor' + str(name) + '/eval')
            self.at_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor' + str(name) + '/target')

            self.soft_replace = [tf.assign(ta, (1-tau)*ta + tau*ea) for ta, ea in zip(self.at_para, self.ae_para)]

            self.opt = tf.train.AdamOptimizer(learning_rate)

    def update_obs(self, state, pool):
        """
        According to global flow and communication pool, agent updates its own local obs.
        :param state: global flow; array, (n_agents,)
        :param pool: communication; array, (n_agents,)
        :return: None
        """
        self.obs = pool
        self.obs[self.name] = state[self.name]

    def get_communicate_action(self):
        """
        Get communication action
        :return: communication action; array, (None, a_dim)
        """
        # (32, ) -> (1, 32)
        action = np.clip(self.sess.run(self.a2, feed_dict={self.input: self.obs[np.newaxis, :]})[0], -1, 1)  # numerical precision
        if self.train:
            action = np.clip(np.random.normal(loc=action, scale=self.var_com), -1, 1)
            self.var_com = self.var_com - self.var_decrement if self.var_com > 0.001 else 0.001
        return action

    def get_cut_action(self):
        """
        Get cut action
        :return: cut action; array, (None, 1)
        """
        action = np.clip(self.sess.run(self.a1, feed_dict={self.input: self.obs[np.newaxis, :]})[0], 0, 1)  # numerical precision
        if self.train:
            action = np.clip(np.random.normal(loc=action, scale=self.var_com), 0, 1)
            self.var_cut = self.var_cut - self.var_decrement if self.var_cut > 0.001 else 0.001
        return action

    def _build_a(self, obs, scope, trainable=False):
        """
        Build actor network
        :param obs: input; array, (None, n_agents)
        :param scope: target/eval; str
        :param trainable: True(eval)/False(target); boolean
        :return: cut_action, com_action
        """

        with tf.variable_scope(scope):

            x = tf.layers.dense(obs, 64, trainable=trainable, name='l1')
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 64, trainable=trainable, name='l2')
            x = tf.nn.relu(x)

            a1 = tf.layers.dense(x, 1, tf.nn.sigmoid, trainable=trainable, name='cut')
            a2 = tf.layers.dense(x, 1, tf.tanh, trainable=trainable, name='com')

        return a1, a2


class Critic:
    def __init__(self, n_agents, sess, actions, actions_, gamma=0.05, tau=0.05, learning_rate=0.002):

        self.sess = sess
        self.gamma = gamma
        self.s_dim = n_agents

        self.s = tf.placeholder(tf.float32, [None, n_agents], 'global_flow')
        self.a = actions
        self.s_ = tf.placeholder(tf.float32, [None, n_agents], 'next_global_flow')
        self.a_ = actions_

        # self.input = tf.concat([self.s, self.a], axis=0, name='s')
        # self.input_ = tf.concat([self.s_, self.a_], axis=0, name='s')

        with tf.variable_scope('Critic'):
            self.q = self._build_c(self.s, self.a, scope='eval', trainable=True)
            self.q_ = self._build_c(self.s_, self.a_, scope='target', trainable=False)

        self.ce_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic'+'/eval')
        self.ct_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic'+'/target')
        self.soft_replace = [tf.assign(tc, (1-tau)*tc+tau*ec) for tc, ec in zip(self.ct_para, self.ce_para)]

        self.opt = tf.train.AdamOptimizer(learning_rate)

    def _build_c(self, s, a, scope, trainable):
        """
        Build critic network
        :param s: global state; array, (None, s_dim)
        :param a: joint actions of cut and com; array, (None, n_agents*2), [[cut_1, com_1], [cut_2, com2]...]
        :param scope: eval/target; str
        :param trainable: True(eval)/False(target); boolean
        :return: q(s,a)
        """

        with tf.variable_scope(scope):
            x = s
            x = tf.layers.dense(x, 64, trainable=trainable)
            x = tf.nn.relu(x)

            x = tf.concat([x, a], axis=-1)

            x = tf.layers.dense(x, 64, trainable=trainable)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
                                trainable=trainable)

        return x


if __name__ == "__main__":
    sess = tf.Session()
    n_agents = 10
    n_attacks = 8

    s = np.random.normal(loc=3, scale=1, size=n_agents)

    agent = DDPG(n_agents=n_agents, sess=sess)

    joint_action = agent.choose_action(s)
    agent.communicate()

    # Get communication count for current step
    # Note each step, max communication count is number of agents.
    print(agent.communicate_counter)

    # Get agent cut action
    agent_id = np.random.choice(n_agents, 3)
    print(joint_action[agent_id])

    # Get communication action
    # It is valid only after agent.communicate() is called
    print(agent.get_agent_communication_action(i=agent_id))  # 3 agents
    print(agent.get_agent_communication_action(i=1))  # agent with id = 1
