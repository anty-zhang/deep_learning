from dqn.maze_env import Maze
from dqn.RL_brain import DeepQNetwork


def run_maze():
    step = 0    # 用来控制什么时候学习
    for episode in range(300):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)   # observation_=[-0.25 -0.5 ]，reward=0

            # 存储记忆
            RL.store_transition(observation, action, reward, observation_)

            # 控制学习时间和频率(先累计一些记忆，再开始学习)
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,  # 每200步替换一次target_net的参数
                      memory_size=2000,     # 记忆上限
                      # output_graph=True   # 是否输出tensorboard文件
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()      # 观看神经网络的误差曲线
