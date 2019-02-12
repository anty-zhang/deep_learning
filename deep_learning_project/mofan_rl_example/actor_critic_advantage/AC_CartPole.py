"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.

The cart pole example. Policy is oscillated.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import numpy as np
import tensorflow as tf
import gym

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

env = gym.make('CartPole-v0')
env.seed(1)  # reproducible
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])         # log 动作概率
            # self.exp_v 为预计的价值
            # log 概率 * TD 方向
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        """
            学习 状态的价值 (state value), 不是行为的价值 (action value),
            计算 TD_error = (r + v_) - v,
            用 TD_error 评判这一步的行为有没有带来比平时更好的结果,
            可以把它看做 Advantage
        """
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error


sess = tf.Session()

actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor

sess.run(tf.global_variables_initializer())

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER:
            env.render()

        a = actor.choose_action(s)

        s_, r, done, info = env.step(a)

        if done:
            r = -20

        track_r.append(r)

        td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            break


"""
问题一：引导Actor更新的是Critic给的td_error,td_error大的话，可以说是下一个状态比现在的状态好，所以能改引导更新，但是td_error小，也不能说明
下一个状态就不好呀，应为td_error是Critic的loss，随着Critic的训练，td_error应该是越来越小才对；

问题二：既然Critic学习的是(或者说输出是)当
前状态的价值，不是应该由Critic的输出self.v来引导Actor的更新吗？随着Critic的收敛，也就是td_error越来越小，self.v(也就是当前状态的打分：
当前状态的价值)就越来越准，所以应该是self.v来引导Actor的更新才对呀

問題一：
td_error不是比較狀態本身好不好(v才是)，而是有多和預期不同(或說''比平時好/壞多少'')
td_error的絕對值大小提供了更新的強度，正負提供方向。在td_error絕對值收斂的前提下，如果td_error絕對值很大，可以想成下一步相當的超乎預期(可能是
超好[td_error為正]，或是超爛[td_error為負])，因此actor學會在該s下，更想輸出該動作或更想避開該動作。
如你所說的，td_error是critic的指標，隨著訓練td_error理論上絕對值要越來越小，代表critic越來越成熟，對環境的評估越來越好，不太有超乎預期的事情。
但環境是複雜或是無法預期的，也許出現了很少發生的狀態，td_error就會很高。

问题二：
提到的思想更像 DDPG (我后面有教程)的思想, DDPG 就是用 Critic 的 Q 估计的梯度来更新 Actor. 所以说 AC 算法可以基于很多种不同的思想. 在一般的
 AC 算法中. 如果直接用 V 来更新 actor, 这种情况经过试验论证, 会比较 bias (每步的更新幅度都不一样), 为了去除这种bias, 我们同时减去一个
 baseline, 具体来说就是, 当前 state 是 v, 因为我们在 v 做了某个动作a达到了下一个状态, 所以下一个state_ 有 r+gamma*v_ 实际上可以认为是
 一个 q(s,a)的值, 用 r+gamma*v_ - v 作为 advantage, 就是在做 q(s, a) - v(s). 这个 advantage 就是 td_error啦, 说的是在这个状态s做
  a 比 在 s 做各个动作的均值相差多少, 如果这个特定动作比各个动作的均值要好, 我们就对 Actor 使用这个 advantage, 往好的方向更新. 用这种相对而
  言的好坏(advantage)学习可以提高收敛性.
"""