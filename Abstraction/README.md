## State Abstraction

It is well-known that state presentation is important for reinforcement learning. A good state presentation can reduce the  difficulty of value approximation. For example, it is neural network that makes reinforcement learning has a strong power and popular again.

However, we can not refute that directly represent state can be redundant from the perspective of learning. Think about that a robot walks in the room and state is presented by location coordinate. There are many state can be clustered into single state, saying inside room and outside room if the target is just determined by these two states. 

Therefore, based on the generative model of markov transition model, we can abstract state from perspective of probability. Specifically, we can transfer primitive state space into low-dimension Euclidean space, upon which close data can be clustered into single class. This is inspired by the [t-SNE](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf).

<div align="center">
   <img src="https://github.com/liziniu/Reinforcement-Learning/blob/master/Abstraction/img/Loss%20Function.png" height="150" width="500">
  <p> Loss Function </p>
</div>

Here is an illustrative example with five states that transition probabilities are given. We can abstract these states into two new states.

<div align="center">
   <img src="https://github.com/liziniu/Reinforcement-Learning/blob/master/Abstraction/img/State%20Abstraction.png" height="400" width="500">
</div>

An experiment is done by the following transition probability matrix, showing that state_1 and state_2 are probabilisticly similar and state_3 and state_4 are pobabilisticly similar as the matrix indicates.

      t_p = np.array([[0.1, 0.8, 0.05, 0.05],
                      [0.8, 0.1, 0.05, 0.05],
                      [0.05, 0.05, 0.1, 0.8],
                       [0.1, 0.1, 0.7, 0.1]])

<div align="center">
   <img src="https://github.com/liziniu/Reinforcement-Learning/blob/master/Abstraction/img/Result.png" height="400" width="500">
</div>
