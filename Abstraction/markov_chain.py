import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


np.random.seed(1)
tf.set_random_seed(1)

# transition probability
t_p = np.array([[0.3, 0.3, 0.2, 0.2],
                [0.2, 0.2, 0.3, 0.3],
                [0.7, 0.1, 0.1, 0.1],
                [0.3, 0.1, 0.4, 0.2]])
# 4 states with 4-dimension
states = np.random.rand(4, 4) * 10
# print(states)

sess = tf.Session()
s = tf.placeholder(tf.float32, [4, 4], name="states")
l1 = tf.layers.dense(s, 20, activation=tf.nn.sigmoid, kernel_initializer=tf.random_normal_initializer(0.3, 1.0))
l2 = tf.layers.dense(l1, 2, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(0.3, 1.0))

opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
sess.run(tf.global_variables_initializer())

# pair-wise distance
res = tf.reshape(tf.tile(l2, [4, 1]), [4, 4, 2]) - tf.expand_dims(l2, 1)

# modify diagonal values
l2_value = sess.run(l2, feed_dict={s: states})
indices = [[0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1], [2, 2, 0], [2, 2, 1], [3, 3, 0], [3, 3, 1]]
values = np.array(l2_value).flatten()
shapes = [4, 4, 2]
delta = tf.SparseTensor(indices, values, shapes)
res = tf.sparse_tensor_to_dense(delta) + res

con_prob = tf.exp(-tf.norm(res, axis=2))
con_prob = con_prob / tf.expand_dims(tf.reduce_sum(con_prob, axis=1), 1)

for i in range(100):
    loss = 0
    for j in range(4):
        loss += tf.reduce_sum(t_p[j] * tf.log(t_p[j]/con_prob[j]))
    if i % 10 == 0:
        print(sess.run(loss, {s: states}))
    sess.run(opt.minimize(loss), {s: states})

result = sess.run(l2, {s: states})
plt.figure()
plt.scatter(result[:, 0], result[:, 1])
plt.show()
