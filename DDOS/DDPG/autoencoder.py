import tensorflow as tf
import numpy as np
from copy import copy
import time

tf.set_random_seed(1)
np.random.seed(1)


class DDPG:
    def __init__(self, n_agents, communication_map, sess, memory_capacity=1e4, batch_size=64, code_length=64,
                 gamma=0.9, noise_type='para',noise_replace_frequency=200, train=True):
        """
        Communication with decoder
        :param n_agents: number of agents; int
        :param communication_map: communication map specify communication destination; array, (n_agents, )
        :param memory_capacity: memory size; int
        :param batch_size: sample batch size; int
        :param gamma: discount; float
        :param code_length: communication code length; int
        :param train: True(train)/False(test); boolean
        :param sess: tensorflow session
        """
        self.n_agents = n_agents
        self.actor_s_dim = n_agents + 1
        self.critic_s_dim = n_agents + 1
        self.gamma = gamma
        self.map = communication_map
        self.code_length = code_length

        self.noise_std = 0.1
        self.noise_type = noise_type
        self.noise_replace_frequency = noise_replace_frequency
        self.adaption_scale = 1.05
        self.desired_distance = 0.15

        # obs is a collection of actor's input
        # pool is a collection of actor's communication code
        self.obs = np.random.uniform(0, 1, [self.n_agents, self.actor_s_dim])
        self.pool = np.zeros(shape=(n_agents, code_length), dtype=np.float32)

        # transition: [s, a, s_, pool, obs, r]
        # s: critic_s_dim; a: n_agents
        # s_: critic_s_dim; pool: n_agents * code_length
        # obs: n_agents * actor_s_dim; r: 1
        self.memory_size = int(memory_capacity)
        self.batch_size = int(batch_size)
        self.memory_pointer = 0
        self.memory = np.zeros(shape=[self.memory_size, self.critic_s_dim*2 + self.n_agents + self.n_agents*self.actor_s_dim+\
                                      self.n_agents*code_length + 1], dtype=np.float32)
        self.train_counter = 0
        self.target_replace = 200

        # build actors and critic
        self.sess = sess
        self.actors = []
        self.actions = []
        self.actions_ = []
        # build centre decoder
        self.decoder = Decoder(code_length=code_length, output_dim=n_agents, sess=sess)
        # build actor
        for i in range(n_agents):
            actor = Actor(name=i, sess=sess, s_dim=self.actor_s_dim, var_decrement=1/self.memory_size,
                          code_length=code_length, decoder=self.decoder, noise_type=noise_type, noise_std=self.noise_std,
                          train=train)
            action = actor.action
            action_ = actor.action_

            self.actors.append(actor)
            self.actions.append(action)
            self.actions_.append(action_)

        a4c = tf.concat(self.actions, axis=1)
        a4c_ = tf.concat(self.actions_, axis=1)
        self.critic = Critic(s_dim=self.critic_s_dim, sess=sess, gamma=gamma, actions=a4c, actions_=a4c_)

        self.soft_replace = [self.critic.soft_replace, self.decoder.soft_replace]
        for i in range(self.n_agents):
            self.soft_replace.append(self.actors[i].soft_replace)
        if self.noise_type == 'para':
            self.noise_distances, self.noise_policy_replaces = [], []
            for actor in self.actors:
                self.noise_distances.append(actor.noise_distance)
                self.noise_policy_replaces.append(actor.noise_policy_replace)

        self.r = tf.placeholder(tf.float32, [None, 1], "reward")

        self.critic_train, self.actor_train, self.decoder_train = self._get_trainer()

        self.saver = tf.train.Saver(max_to_keep=1)

        if train:
            self.sess.run(tf.global_variables_initializer())
        else:
            model_file = tf.train.latest_checkpoint('ckpt-autoencoder/')
            self.saver.restore(sess, model_file)

    def communicate(self):
        """
        Each agent pass its own embedding as code to destination agent
        :return: None
        """
        for i in range(self.n_agents):
            actor = self.actors[i]
            des = self.map[i]
            code_send = actor.send_code()
            self.pool[des] = code_send

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
            actor.receive_obs(state[i], state[-1])    # (local flow, global flow)

            code_receive = self.pool[i]
            obs = actor.receive_communication(code_receive)
            self.obs[i] = obs

            actions[i] = actor.get_action(net='eval')
        # update noise_std
        if self.memory_pointer > self.batch_size and self.memory_pointer % self.noise_replace_frequency == 0 and self.noise_type == 'para':
            self._adaptive_noise_std()
        return actions

    def store_transition(self, s, a, r, s_):
        """
        Store experience
        :param s: global flow state; array, (n_agents, )
        :param s_: next global flow, state; array, (n_agents, )
        :param a: joint cut action; array, (n_agents, )
        :param r: joint reward; float
        :return: None
        """
        # Note sample include: s, a, s_, pool, obs, r
        t = np.hstack([s, a, s_, self.pool.flatten().copy(), self.obs.flatten().copy(), [r]])
        index = int(self.memory_pointer % self.memory_size)
        self.memory[index, :] = t
        self.memory_pointer += 1

    def learn(self):
        """
        Sample from memory and Train actors and critic
        :return: None
        """
        if self.train_counter % self.target_replace == 0:
            self.sess.run(self.soft_replace)

        s, a, s_, pool, obs, r = self._sample()
        # actor and decoder feed_dict
        actor_feed = {}
        actor_feed[self.critic.s] = s
        # critic feed_dict
        critic_feed = {}
        critic_feed[self.critic.s] = s
        critic_feed[self.critic.s_] = s_
        critic_feed[self.r] = r[:, np.newaxis]
        critic_feed[self.critic.a] = a

        # reconstruct a_ for critic
        # a_ <- obs_  <- s_
        #             <- code  <- pool
        a_ = []   # for critic
        decoder_input = []
        decoder_output = []
        for i in range(self.n_agents):
            actor = self.actors[i]

            code_receive = copy(pool[:, i*self.code_length:(i+1)*self.code_length])
            code_decode = self.decoder.decode(code_receive, net='target')

            actor_obs = copy(obs[:, i*self.actor_s_dim:(i+1)*self.actor_s_dim])
            actor_obs_ = copy(s_)
            actor_obs_[:, :-1] = code_decode
            actor_obs_[:, i] = s_[:, i]

            # critic
            actor_action_ = actor.get_action(net='target', obs=actor_obs_)     # (batch_size, 1)
            a_.append(actor_action_)

            # decoder
            decoder_input.append(code_receive)
            decoder_output.append(s_[:, :-1])

            # actor
            actor_feed[actor.input] = actor_obs
        a_ = np.hstack(a_)   # (batch_size, n_agents)
        critic_feed[self.critic.a_] = a_

        decoder_feed = {}
        decoder_batch_size = self.n_agents * 10
        decoder_batch_indices = np.random.choice(len(decoder_input), decoder_batch_size)
        decoder_feed[self.decoder.input] = np.vstack(decoder_input)[decoder_batch_indices, :]
        decoder_feed[self.decoder.state] = np.vstack(decoder_output)[decoder_batch_indices, :]

        # train
        # t1 = time.time()
        self.sess.run(self.critic_train, feed_dict=critic_feed)
        # t2 = time.time()
        self.sess.run(self.actor_train, feed_dict=actor_feed)
        # t3 = time.time()
        self.sess.run(self.decoder_train, feed_dict=decoder_feed)
        # t4 = time.time()
        # print("critic:", t2-t1, "actor: ", t3-t2, "decoder: ", t4-t3)
        self.train_counter += 1

    def save_model(self):
        """
        Save model parameters
        :return: None
        """
        self.saver.save(self.sess, 'ckpt-autoencoder/ddpg.ckpt')

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
        # transition: [s, a, s_, pool, obs, r]
        s = batch[:, :self.critic_s_dim]
        a = batch[:, self.critic_s_dim: self.critic_s_dim+self.n_agents]
        s_ = batch[:, self.critic_s_dim+self.n_agents: self.critic_s_dim*2+self.n_agents]
        pool = batch[:, self.critic_s_dim*2+self.n_agents: self.critic_s_dim*2+self.n_agents+self.n_agents*self.code_length] # code at s_
        obs = batch[:, self.critic_s_dim*2+self.n_agents+self.n_agents*self.code_length:-1] # input for actor network; (n_agents, n_agents)
        r = batch[:, -1]

        return s, a, s_, pool, obs, r

    def _adaptive_noise_std(self):
        s, a, s_, pool, obs, r = self._sample(update_noise=True)

        feed_dict = {}
        for i in range(self.n_agents):
            actor = self.actors[i]
            actor_obs = copy(obs[:, i*self.actor_s_dim:(i+1)*self.actor_s_dim])
            feed_dict[actor.input] = actor_obs

        distances = self.sess.run(self.noise_distances, feed_dict=feed_dict)

        for i, d in enumerate(distances):
            if d < self.desired_distance:
                self.actors[i].noise_std = self.actors[i].noise_std * self.adaption_scale
            else:
                self.actors[i].noise_std = self.actors[i].noise_std / self.adaption_scale

        self.sess.run(self.noise_policy_replaces)


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
        return critic_train, actor_train, self.decoder.train

class Actor:
    def __init__(self, name, sess, s_dim, a_dim=1, code_length=64, tau=0.01, var=0.5, var_decrement=1e-4,
                 learning_rate=1e-4, decoder=None, noise_type='para', noise_std=0.1, train=True):
        """
        Single Actor
        :param name: identity; int
        :param sess: tensorflow session
        :param s_dim: state's dim; int
        :param a_dim: cut action's dim; int
        :param code_length: code length(embedding length); int
        :param tau: ratio of replace target; float
        :param var: variance of cut action; float
        :param train: True(train)/False(test); boolean
        :param learning_rate: learning rate of optimizer; float
        """
        self.a_dim = a_dim
        self.name = name
        self.sess = sess
        self.action_noise_std = var
        self.action_noise_decrement = var_decrement
        self.train = train

        self.decoder = decoder
        self.local_flow = 0
        self.global_flow = 0
        self.obs = np.zeros(s_dim)
        self.noise_type =noise_type

        # with graph.as_default():
        with tf.variable_scope('Actor' + str(name)):
            self.input = tf.placeholder(tf.float32, [None, s_dim], 'obs')
            self.input_ = tf.placeholder(tf.float32, [None, s_dim], 'obs_')

            self.action, self.code = self._build_a(self.input, code_length, scope='eval', trainable=True)
            self.action_, _ = self._build_a(self.input_, code_length, scope='target', trainable=False)
            if self.noise_type == 'para':
                self.action_noise, self.code_noise = self._build_a(self.input, code_length, scope='noise', trainable=True)
                self.noise_std = noise_std

            self.ae_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor' + str(name) + '/eval')
            self.at_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor' + str(name) + '/target')

            self.soft_replace = [tf.assign(ta, (1-tau)*ta + tau*ea) for ta, ea in zip(self.at_para, self.ae_para)]
            if self.noise_type == 'para':
                self.an_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor' + str(name) + '/noise')
                self.noise_policy_replace = [tf.assign(an, ae + tf.random_normal(tf.shape(ae), mean=0.0, stddev=self.noise_std))
                                             for ae, an in zip(self.ae_para, self.an_para)]
                self.noise_distance = tf.sqrt(tf.reduce_mean(tf.square(self.action-self.action_noise)))

            self.opt = tf.train.RMSPropOptimizer(learning_rate)

    def receive_obs(self, local_flow, global_flow):
        """
        Receive and Update its own flow info
        :param local_flow: its own flow at current step; float
        :param global_flow: global flow at current step; float
        :return: None
        """
        self.local_flow = local_flow
        self.global_flow = global_flow

    def receive_communication(self, code):
        """
        Receive code from other agent and construct actor network input
        :param code: code from others; array, (code_length, )
        :return: obs(actor network input); array, (s_dim, )
        """
        self.obs[:-1] = self.decoder.decode(code, net='eval')
        self.obs[self.name] = self.local_flow
        self.obs[-1] = self.global_flow
        return self.obs

    def send_code(self):
        """
        Send code to destination agent
        :return: code; array, (code_length, )
        """
        return self.sess.run(self.code, feed_dict={self.input: self.obs[np.newaxis, :]})[0]

    def get_action(self, net='eval', obs=None):
        """
        Get cut flow action
        :param net: 'eval'(choose action)/'target'(train); str
        :param obs: actor network input; array, (n_samples, s_dim)
        :return:
        """
        action = None
        if net == 'eval':
            action = np.clip(self.sess.run(self.action, feed_dict={self.input: self.obs[np.newaxis, :]})[0], 0, 1)
            if self.train:
                if self.noise_type == 'action':
                    action = self.sess.run(self.action, feed_dict={self.input: self.obs[np.newaxis, :]})[0]
                    action = np.clip(np.random.normal(loc=action, scale=self.action_noise_std), 0, 1)
                    self.action_noise_std = self.action_noise_std - self.action_noise_decrement if self.action_noise_std > 0.001 else 0.001
                if self.noise_type == 'para':
                    action = self.sess.run(self.action_noise, feed_dict={self.input: self.obs[np.newaxis, :]})[0]
        if net == 'target':
            if obs is None:
                raise ValueError("obs should be given for target network")
            action = self.sess.run(self.action_, feed_dict={self.input_: obs})  # multi samples
        return action

    def _build_a(self, obs, code_length, scope, trainable=False):
        """
        Build actor network
        :param obs: input; array, (None, n_agents)
        :param scope: target/eval; str
        :param trainable: True(eval)/False(target); boolean
        :return: cut_action; code;
        """
        with tf.variable_scope(scope):
            x = tf.layers.dense(obs, 64, trainable=trainable, name='l1')
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, code_length, trainable=trainable, name='l2')
            code = tf.nn.relu(x)

            a = tf.layers.dense(code, 1, tf.nn.sigmoid, trainable=trainable, name='cut_action')

        return a, code


class Critic:
    def __init__(self, s_dim, sess, actions, actions_, gamma=0.9, tau=0.01, learning_rate=1e-3):
        """
        Critic network
        :param s_dim: dim of global info; int
        :param sess: TensorFlow session
        :param actions: agent cut actions; array/tensor, (None, n_agents)
        :param actions_: agent next cut actions; array/tensor, (None, n_agents)
        :param gamma: reward discount ratio; float
        :param tau: replace ratio; float
        :param learning_rate: learning rate; float
        """
        self.sess = sess
        self.gamma = gamma
        self.s_dim = s_dim

        self.s = tf.placeholder(tf.float32, [None, s_dim], 'global_flow')
        self.a = actions
        self.s_ = tf.placeholder(tf.float32, [None, s_dim], 'next_global_flow')
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


class Decoder:
    def __init__(self, code_length, output_dim, sess, lr=0.005, tau=0.05):
        """
        Decoder embedded in actor
        :param name: id; int
        :param code_length: code length; int
        :param s_dim: actor input dim; int
        :param sess: TensorFlow session
        :param lr: learning rate; float
        :param tau: replace ratio; float
        """
        self.input = tf.placeholder(tf.float32, [None, code_length], "code")
        self.input_ = tf.placeholder(tf.float32, [None, code_length], "code_")
        self.state = tf.placeholder(tf.float32, [None, output_dim], "state_info")

        with tf.variable_scope('Decoder'):
            self.output = self._build_decoder(self.input, output_dim, scope='Eval', trainable=True)
            self.output_ = self._build_decoder(self.input_, output_dim, scope='Target', trainable=False)

        self.eva_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Decoder' + '/Eval')
        self.tar_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Decoder' + '/Target')

        self.soft_replace = [tf.assign(tc, (1 - tau) * tc + tau * ec) for tc, ec in zip(self.tar_para, self.eva_para)]

        self.sess = sess
        self.loss = tf.losses.mean_pairwise_squared_error(labels=self.state, predictions=self.output)
        self.train = tf.train.RMSPropOptimizer(lr).minimize(self.loss, var_list=self.eva_para)

    def decode(self, code, net):
        """
        Decode code
        :param code: code; array, (None, code_length)
        :param net: 'eval'(choose action)/'target'(train); str
        :return: code; array, (None, actor_network_input_dim)
        """
        if net == 'eval':
            return self.sess.run(self.output, feed_dict={self.input: code[np.newaxis, :]})[0]  # single sample
        if net == 'target':
            return self.sess.run(self.output_, feed_dict={self.input_: code})   # multi sample

    def _build_decoder(self, inputs, output_dim, scope, trainable):
        with tf.variable_scope(scope):
            x = tf.layers.dense(inputs, 64, trainable=trainable)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 64, trainable=trainable)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, output_dim, trainable=trainable)
        return x

if __name__ == "__main__":
    sess = tf.Session()
    n_agents = 2
    n_attacks = 8

    s = np.zeros(n_agents+1)
    s[:-1] = np.random.normal(loc=3, scale=1, size=n_agents)
    s[-1] = sum(s[:-1])

    agent = DDPG(n_agents=n_agents, communication_map=np.array([1, 0]), sess=sess, memory_capacity=1, batch_size=1, code_length=2)

    joint_action = agent.choose_action(s)
    agent.communicate()

    agent.store_transition(s, joint_action, 1, s+1)

    agent.learn()


