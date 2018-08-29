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

An experiment result based on above algorithm is as follows. In the future, I will enlarge the size of problem.

<div align="center">
   <img src="https://github.com/liziniu/Reinforcement-Learning/blob/master/Abstraction/img/Result.png" height="400" width="500">
</div>
