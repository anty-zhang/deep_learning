"""
Deep Q network,

Using:
Tensorflow: 1.0
gym: 0.7.3
"""


import gym
from openaigym.RL_brain import DeepQNetwork

env = gym.make('CartPole-v0')       # 定义使用gym库中的一个环境
env = env.unwrapped                 # 不做这个会有很多限制

print(env.action_space)             # 环境中可用的action数量
print(env.observation_space)        # 环境中的的state的observation有多少个
print(env.observation_space.high)   # 查看observation最高取值
print(env.observation_space.low)    # 查看observation最低取值

RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001,)

total_steps = 0


for i_episode in range(100):

    observation = env.reset()
    ep_r = 0
    while True:
        env.render()                                            # 刷新环境

        action = RL.choose_action(observation)                  # 选行为

        observation_, reward, done, info = env.step(action)     # 获取下一个state

        # x 是车的水平位移, 所以 r1 是车越偏离中心, 分越少
        # theta 是棒子离垂直的角度, 角度越大, 越不垂直. 所以 r2 是棒越垂直, 分越高
        # the smaller theta and closer to center the better
        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        reward = r1 + r2        # 总 reward 是 r1 和 r2 的结合, 既考虑位置, 也考虑角度, 这样 DQN 学习更有效率

        RL.store_transition(observation, action, reward, observation_)

        ep_r += reward
        if total_steps > 1000:
            RL.learn()

        if done:
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1

RL.plot_cost()
