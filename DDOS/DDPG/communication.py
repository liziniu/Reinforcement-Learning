import tensorflow as tf
import numpy as np
from copy import copy

np.random.seed(1)
tf.set_random_seed(1)


class DDPG:
    def __init__(self, n_agents, sess, memory_capacity=1e4, batch_size=64, gamma=0.9, noise_type='para',
                 noise_replace_frequency=200, train=True):
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
        self.actor_s_dim = n_agents + 1
        self.critic_s_dim = n_agents + 1
        self.gamma = gamma
        self.target_replace = 200
        self.noise_replace_frequency = noise_replace_frequency

        # communication pool, np.array, shape:(n_agents, )
        self.pool = np.clip(np.random.normal(loc=0, scale=0.3, size=n_agents), -1, 1)
        self.pool_copy = self.pool.copy()
        self.communicate_counter = 0
        self.train_counter = 0
        # transition:[state, pool, state_, pool_, action, reward]
        # state: n_agents+1; pool: n_agents
        # state_: n_agents+1; pool_: n_agents
        # action: n_agents; reward: 1
        self.memory_size = int(memory_capacity)
        self.batch_size = int(batch_size)
        self.memory_pointer = 0
        self.memory = np.zeros(shape=[self.memory_size, self.critic_s_dim*2+self.n_agents*3+1], dtype=np.float32)

        # noise
        self.noise_type = noise_type
        self.noise_std = 0.1
        if self.noise_type == 'para':
            self.adaption_scale = 1.05
            self.desired_distance = 0.15

        self.sess = sess
        self.actors = []
        self.actions = []
        self.actions_ = []

        for i in range(n_agents):
            actor = Actor(name=i, sess=sess, s_dim=self.actor_s_dim, var_decrement=1/memory_capacity,
                          noise_type=self.noise_type, noise_std=self.noise_std, train=train)
            action = tf.concat([actor.a1, actor.a2], axis=1)
            action_ = tf.concat([actor.a1_, actor.a2_], axis=1)

            self.actors.append(actor)
            self.actions.append(action)
            self.actions_.append(action_)

        a4c = tf.concat(self.actions, axis=1)
        a4c_ = tf.concat(self.actions_, axis=1)
        self.critic = Critic(s_dim=self.critic_s_dim, sess=sess, gamma=gamma, actions=a4c, actions_=a4c_)

        # target network soft replacement
        self.replace_target = [actor.soft_replace for actor in self.actors]
        self.replace_target.append(self.critic.soft_replace)

        # distance
        if self.noise_type == 'para':
            self.noise_distances, self.noise_policy_replaces = [], []
            for actor in self.actors:
                self.noise_distances.append(actor.noise_distance)
                self.noise_policy_replaces.append(actor.noise_policy_replace)

        self.r = tf.placeholder(tf.float32, [None, 1], "reward")

        self.critic_train, self.actor_train = self._get_trainer()

        self.saver = tf.train.Saver(max_to_keep=1)

        if train:
            self.sess.run(tf.global_variables_initializer())
        else:
            model_file = tf.train.latest_checkpoint('ckpt-communication/')
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
            if communicate_action >= 0:
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
            actor.update_obs(state.copy(), self.pool.copy())
            actions[i] = actor.get_cut_action()
        # update noise std
        if self.memory_pointer > self.batch_size and self.memory_pointer % self.noise_replace_frequency == 0 and self.noise_type=='para':
            self._adaptive_noise_std()
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
        if self.train_counter % self.target_replace == 0:
            self.sess.run(self.replace_target)

        s, pool, s_, pool_, cut_a, r = self._sample()

        # reconstruct actions for critic from pool_ and cut_a
        a = np.concatenate([cut_a, pool_], axis=1)

        critic_feed = {}
        critic_feed[self.critic.s] = s
        critic_feed[self.critic.s_] = s_
        critic_feed[self.r] = r[:, np.newaxis]    # (32) - > (32, 1)
        critic_feed[self.critic.a] = a

        actor_feed = {}
        actor_feed[self.critic.s] = s
        # 恢复obs
        for i in range(self.n_agents):
            obs = copy(s)
            obs[:, :-1] = copy(pool)
            obs[:, i] = s[:, i]

            obs_ = copy(s_)
            obs_[:, :-1] = copy(pool_)
            obs_[:, i] = s_[:, i]

            # the process of critic's train does not need actor eval network(obs),
            # but it need obs_ to get a_ for target q network
            critic_feed[self.actors[i].input_] = obs_
            # update actor network does not need target network
            actor_feed[self.actors[i].input] = obs

        # critic
        self.sess.run(self.critic_train, feed_dict=critic_feed)
        self.sess.run(self.actor_train, feed_dict=actor_feed)
        self.train_counter += 1

    def save_model(self):
        """
        Save model parameters
        :return: None
        """
        self.saver.save(self.sess, 'ckpt-communication/ddpg.ckpt')

    def _adaptive_noise_std(self):
        s, pool, s_, pool_, cut_a, r = self._sample(update_noise=True)

        feed_dict = {}
        for i in range(self.n_agents):
            obs = copy(s)
            obs[:, :-1] = copy(pool)
            obs[:, i] = s[:, i]
            feed_dict[self.actors[i].input] = obs

        distances = self.sess.run(self.noise_distances, feed_dict=feed_dict)

        for i, d in enumerate(distances):
            if d < self.desired_distance:
                self.actors[i].noise_std = self.actors[i].noise_std * self.adaption_scale
            else:
                self.actors[i].noise_std = self.actors[i].noise_std / self.adaption_scale

        self.sess.run(self.noise_policy_replaces)

    def _sample(self, update_noise=False):
        if self.memory_pointer < self.memory_size:
            if update_noise:
                start = min(self.memory_pointer - self.batch_size, int(self.memory_pointer/2))
                end = self.memory_pointer
                indices = np.random.choice(np.arange(start, end), self.batch_size)
            else:
                indices = np.random.choice(self.memory_pointer, self.batch_size)
        else:
            if update_noise:
                start = (self.memory_pointer - self.noise_replace_frequency * 10) % self.memory_size
                end = self.memory_pointer % self.memory_size
                indices = np.random.choice(np.arange(start, end), self.batch_size)
            else:
                indices = np.random.choice(self.memory_size, self.batch_size)
        batch = self.memory[indices, :]
        s = batch[:, :self.critic_s_dim]
        pool = batch[:, self.critic_s_dim: self.n_agents+self.critic_s_dim]
        s_ = batch[:, self.n_agents+self.critic_s_dim: self.n_agents+self.critic_s_dim*2]
        pool_ = batch[:, self.n_agents+self.critic_s_dim*2: self.n_agents*2 + self.critic_s_dim*2]
        cut_a = batch[:, self.n_agents*2 + self.critic_s_dim*2: self.n_agents*3 + self.critic_s_dim*2]
        r = batch[:, -1]

        return s, pool, s_, pool_, cut_a, r

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
    def __init__(self, name, sess, s_dim, a_dim=1, tau=0.01, var_cut=0.9, var_com=0.9, var_decrement=1e-4, noise_std=0.1,
                 desired_distance=0.15, learning_rate=1e-4, noise_type='action', train=True):
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
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.desired_distance = desired_distance
        self.train = train

        # with graph.as_default():
        with tf.variable_scope('Actor' + str(name)):
            self.input = tf.placeholder(tf.float32, [None, s_dim], 'obs')
            self.input_ = tf.placeholder(tf.float32, [None, s_dim], 'obs_')
            self.obs = np.zeros(s_dim, np.float32)

            self.a1, self.a2 = self._build_a(self.input, scope='eval', trainable=True)
            self.a1_, self.a2_ = self._build_a(self.input_, scope='target', trainable=False)
            if self.noise_type == 'para':
                self.a1_noise, self.a2_noise = self._build_a(self.input, scope='noise', trainable=False)
                self.an_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor' + str(name) + '/noise')
            self.ae_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor' + str(name) + '/eval')
            self.at_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor' + str(name) + '/target')

            self.soft_replace = [tf.assign(ta, (1-tau)*ta + tau*ea) for ta, ea in zip(self.at_para, self.ae_para)]
            if self.noise_type == 'para':
                self.noise_policy_replace = [tf.assign(an, ae + tf.random_normal(tf.shape(ae), mean=0.0, stddev=self.noise_std))
                                             for ae, an in zip(self.ae_para, self.an_para)]

                self.noise_distance = tf.sqrt(tf.reduce_mean(tf.square(self.a1-self.a1_noise) +
                                                             tf.square(self.a2-self.a2_noise)))
            self.opt = tf.train.RMSPropOptimizer(learning_rate)

    def update_obs(self, state, pool):
        """
        According to global flow and communication pool, agent updates its own local obs.
        :param state: global flow; array, (n_agents,)
        :param pool: communication; array, (n_agents,)
        :return: None
        """
        self.obs = copy(state)
        self.obs[: -1] = pool
        self.obs[self.name] = state[self.name]

    def get_communicate_action(self):
        """
        Get communication action
        :return: communication action; array, (None, a_dim)
        """
        action = None
        if self.train:
            if self.noise_type == 'action':
                action = self.sess.run(self.a2, feed_dict={self.input: self.obs[np.newaxis, :]})[0]
                action = np.clip(np.random.normal(loc=action, scale=self.var_com), -1, 1)
                self.var_com = self.var_com - self.var_decrement if self.var_com > 0.001 else 0.001
            if self.noise_type == 'para':
                action = self.sess.run(self.a2_noise, feed_dict={self.input: self.obs[np.newaxis, :]})[0]
        else:
            action = self.sess.run(self.a2, feed_dict={self.input: self.obs[np.newaxis, :]})[0]
        return action

    def get_cut_action(self):
        """
        Get cut action
        :return: cut action; array, (None, 1)
        """
        action = None
        if self.train:
            if self.noise_type == 'action':
                action = self.sess.run(self.a1, feed_dict={self.input: self.obs[np.newaxis, :]})[0]
                action = np.clip(np.random.normal(loc=action, scale=self.var_com), 0, 1)
                self.var_cut = self.var_cut - self.var_decrement if self.var_cut > 0.001 else 0.001
            if self.noise_type == 'para':
                action = self.sess.run(self.a1_noise, feed_dict={self.input: self.obs[np.newaxis, :]})[0]
        else:
            action = self.sess.run(self.a1, feed_dict={self.input: self.obs[np.newaxis, :]})[0]
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
    def __init__(self, s_dim, sess, actions, actions_, gamma=0.9, tau=0.01, learning_rate=1e-3):

        self.sess = sess
        self.gamma = gamma
        self.s_dim = s_dim

        self.s = tf.placeholder(tf.float32, [None, self.s_dim], 'global_flow')
        self.a = actions
        self.s_ = tf.placeholder(tf.float32, [None, self.s_dim], 'next_global_flow')
        self.a_ = actions_

        # self.input = tf.concat([self.s, self.a], axis=0, name='s')
        # self.input_ = tf.concat([self.s_, self.a_], axis=0, name='s')

        with tf.variable_scope('Critic'):
            self.q = self._build_c(self.s, self.a, scope='eval', trainable=True)
            self.q_ = self._build_c(self.s_, self.a_, scope='target', trainable=False)

        self.ce_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic'+'/eval')
        self.ct_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic'+'/target')
        self.soft_replace = [tf.assign(tc, (1-tau)*tc+tau*ec) for tc, ec in zip(self.ct_para, self.ce_para)]

        self.opt = tf.train.RMSPropOptimizer(learning_rate)

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
    n_agents = 2
    n_attacks = 8

    s = np.zeros(n_agents+1)
    s[:-1] = np.random.normal(loc=3, scale=1, size=n_agents)
    s[-1] = sum(s[:-1])

    agent = DDPG(n_agents=n_agents, sess=sess, memory_capacity=1, batch_size=1)

    joint_action = agent.choose_action(s)
    agent.communicate()

    agent.store_transition(s, joint_action, 1, s+1)

    agent.learn()
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
