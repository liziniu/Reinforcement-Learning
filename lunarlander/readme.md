## Self-paced DQN

[Prioritized Experience Replay(PER)](https://arxiv.org/abs/1511.05952) is a recently proposed training regime inspired by large td-error samples can effectively back-propagate and thus accelerate learning.

However, it is a fact that approximate value function cannot converge theoretically and there is overestimate in q-learning. My doubt is that PER can result in oscillation at the beginning of train and introduced additional error for value approximation can harm the subsequent learning.

Motived from [self-paced learning](https://papers.nips.cc/paper/5568-self-paced-learning-with-diversity), a training regime that learning from easy samples to complex samples, oscillation and overestimate in PER can be alleviated. An illustrative example is as follows. With anomalous data, a traditional classification like SVM is easily affected under uniform sampling while self-paced learning can be more robust because anomalous data is added into learning only at the end of learning, upon which its effect of false gradients shift is small.


<div align="center">
  <img src="https://github.com/liziniu/Reinforcement-Learning/blob/master/lunarlander/pic/example.png" height="500" width="500">
</div>


The following is an experiment result based on gym, showing that PER, sensitive to random seed, may be worse than original deep q-learning and self-paced deep q-learning is more robust and high efficient.

<div align="center">
  <img src="https://github.com/liziniu/reinforcement_learning/blob/master/lunarlander/pic/6431523955071_.pic_hd.png" height="300" width="500">
</div>

<div align="center">
  <img src="https://github.com/liziniu/reinforcement_learning/blob/master/lunarlander/pic/Figure_1.png" height="300" width="500">
</div>
