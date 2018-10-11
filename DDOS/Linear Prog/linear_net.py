import tensorflow as tf
import numpy as np

np.random.seed(1)
tf.set_random_seed(1)


class LinearNet:
    def __init__(self, n_agents, lr=1e-3, sess=None):
        self.s_dim = n_agents

        self.S = tf.placeholder(tf.float32, [None, self.s_dim], 's')
        self.A_refer = tf.placeholder(tf.float32, [None, self.s_dim], 'a')

        self.a = self._build_net(self.S)

        loss = tf.reduce_mean(tf.square(self.a - self.A_refer))

        self.train = tf.train.RMSPropOptimizer(lr).minimize(loss)

        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        """
        :param s: (n_agents, )
        :return:
        """
        if len(s.shape) == 1:
            a = self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
        else:
            a = self.sess.run(self.a, {self.S: s[np.newaxis, :]})
        return a

    def _build_net(self, s):
        x = tf.layers.dense(s, 64)
        x = tf.nn.relu(x)

        x = tf.layers.dense(x, 64)
        x = tf.nn.relu(x)

        a = tf.nn.sigmoid(x, self.s_dim)

        return a

    def learn(self, s, a_refer):
        """
        :param s: (n_samples, n_agents)
        :param a_refer: (n_samples, n_agents)
        :return:
        """
        self.sess.run(self.train, feed_dict={self.S: s, self.A_refer: a_refer})