"""
Centralized DDPG with multi-thread for actor update
"""
import tensorflow as tf
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)


class DDPG:
    def __init__(self, n_agents, var=0.9, lr_a=1e-4, lr_c=1e-3, tau=0.01, gamma=0.9, memory_capacity=1e4,
                 batch_size=64, noise_type='para', noise_replace_frequency=200, train=True, sess=None):
        """
        Centralized DDPG
        :param n_agents: number of agents; int
        :param var: action variance; float
        :param lr_a: learning rate of actor; float
        :param lr_c: learning rate of critic; float
        :param tau: soft replace ratio; float
        :param gamma: reward discount ratio; float
        :param memory_capacity: memory size; int
        :param batch_size: sample batch size; int
        :param train: True(train)/False(test); boolean
        :param sess: tensorflow session
        """
        self.var = var
        self.a_dim = n_agents
        self.s_dim = n_agents + 1
        self.memory_capacity = int(memory_capacity)
        self.batch_size = batch_size
        self.train = train
        self.train_counter = 0
        self.noise_type = noise_type
        self.noise_replace_frequency = noise_replace_frequency
        self.target_replace_frequency = 200

        self.memory = np.zeros((int(memory_capacity), self.s_dim * 2 + self.a_dim + 1), dtype=np.float32)
        self.memory_pointer = 0

        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess

        self.S = tf.placeholder(tf.float32, [None, self.s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, self.s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
            # noise policy network
            if self.noise_type == 'para':
                self.noise_std = 0.1
                self.adaption_scale = 1.05
                self.desired_distance = 0.15
                self.a_noise = self._build_a(self.S, scope='noise', trainable=False)
                self.an_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/noise')
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - tau) * ta + tau * ea), tf.assign(tc, (1 - tau) * tc + tau * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        if self.noise_type == 'para':
            self.noise_policy_replace = [tf.assign(an, ae + tf.random_normal(tf.shape(ae), mean=0.0, stddev=self.noise_std))
                                         for ae, an in zip(self.ae_params, self.an_params)]
            self.noise_distance = tf.sqrt(tf.reduce_mean(tf.square(self.a_noise - self.a)))

        q_target = self.R + gamma * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(lr_c).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(lr_a).minimize(a_loss, var_list=self.ae_params)

        self.saver = tf.train.Saver(max_to_keep=1)
        if train:
            self.sess.run(tf.global_variables_initializer())
        else:
            model_file = tf.train.latest_checkpoint('ckpt-ddpg/')
            self.saver.restore(sess, model_file)

    def choose_action(self, s):
        """
        choose joint cut action
        :param s: global flow state; array, (n_agents, )
        :return: joint cut action; array, (n_agents, )
        """
        a = None
        if self.train:
            if self.noise_type == 'action':
                a = self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
                a = np.clip(np.random.normal(loc=a, scale=self.var), a_min=0, a_max=1.0)
                self.var = self.var - 1/self.memory_capacity if self.var > 0.001 else 0.001
            if self.noise_type == 'para':
                a = self.sess.run(self.a_noise, {self.S: s[np.newaxis, :]})[0]
                if self.memory_pointer > self.batch_size and self.memory_pointer % self.noise_replace_frequency == 0:
                    self._adaptive_noise_std()
        else:
            a = self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
        return a

    def learn(self):
        """
        Sample from memory and learn
        :return: None
        """
        # soft target replacement
        if self.train_counter % self.target_replace_frequency== 0:
            self.sess.run(self.soft_replace)

        if self.memory_pointer < self.memory_capacity:
            indices = np.random.choice(self.memory_pointer, self.batch_size)
        else:
            indices = np.random.choice(self.memory_capacity, self.batch_size)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

        self.train_counter += 1

    def store_transition(self, s, a, r, s_):
        """
        Store transition
        :param s: global flow; array, (n_agents, )
        :param a: joint cut action; array, (n_agent, )
        :param r: joint reward; float
        :param s_: next global flow; array, (n_agents, )
        :return: None
        """
        transition = np.hstack((s, a, [r], s_))
        indices = self.memory_pointer % self.memory_capacity
        self.memory[indices, :] = transition
        self.memory_pointer += 1

    def save_model(self):
        self.saver.save(self.sess, 'ckpt-ddpg/ddpg.ckpt')

    def _build_a(self, s, scope, trainable):
        """
        Build actor network
        :param s: global flow; array, (n_samples, n_agents)
        :param scope: eval/target; str
        :param trainable: True(eval)/False(Target); boolean
        :return: joint action; array, (n_sample, n_agents)
        """
        with tf.variable_scope(scope):
            # net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            # a = tf.layers.dense(net, self.a_dim, activation=tf.nn.sigmoid, name='a', trainable=trainable)
            # return a
            x = s
            x = tf.layers.dense(x, 64, trainable=trainable)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 64, trainable=trainable)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, self.a_dim, trainable=trainable)
            x = tf.nn.sigmoid(x)

        return x

    def _build_c(self, s, a, scope, trainable):
        """
        Build critic network
        :param s: global flow; array, (n_samples, n_agents)
        :param a: joint action; array, (n_samples, n_agents)
        :param scope: eval/target; str
        :param trainable: True(eval)/False(Target); boolean
        :return: Q(s, a)
        """
        with tf.variable_scope(scope):
            x = tf.layers.dense(s, 64, trainable=trainable)
            x = tf.nn.relu(x)

            x = tf.concat([x, a], axis=-1)

            x = tf.layers.dense(x, 64, trainable=trainable)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3), trainable=trainable)
        return x

    def _adaptive_noise_std(self):
        if self.memory_pointer < self.memory_capacity:
            indices = np.random.choice(self.memory_pointer, self.batch_size)
        else:
            indices = np.random.choice(self.memory_capacity, self.batch_size)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]

        distance = self.sess.run(self.noise_distance, feed_dict={self.S: bs})

        if distance > self.desired_distance:
            self.noise_std = self.noise_std / self.adaption_scale
        else:
            self.noise_std = self.noise_std * self.adaption_scale

        self.sess.run(self.noise_policy_replace)


if __name__ == "__main__":
    # If you want to get the action of attackers, just specify the indices of attackers.
    n_agents = 108
    n_attacks = 8
    agent = DDPG(n_agents)
    s = np.random.normal(0, 1, n_agents)

    # print attackers action
    attack_id = np.random.choice(n_agents, n_attacks)   # shape: (8, )
    print(agent.choose_action(s)[attack_id])      # shape: (8, )
