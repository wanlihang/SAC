import sys
import time

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from visualdl import LogWriter

from ActorCriticModel import ActorCriticModel
from DE import DE
from Memory import Memory
from SAC import SAC
from SACAgent import SACAgent
from env import Environment


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_off_policy_agent(env, action_dim, sac_agent, epochs, memory, warmup_steps, batch_size):
    episode_reward_list = []
    max_episode_reward = sys.float_info.min
    learn_steps = 0
    # Initialize population
    population = []
    for i in range(10):
        with tqdm(total=int(epochs / 10), desc='Iteration %d' % i) as pbar:
            for epoch in range(int(epochs / 10)):
                # 状态初始化
                state = env.reset()
                # 智能体与环境交互一个回合的回合总奖励
                episode_reward = 0
                # 回合开始
                for time_step in range(200):
                    learn_steps += 1
                    if memory.size() < warmup_steps:
                        action = np.random.uniform(-1, 1, size=action_dim)
                    else:
                        action = sac_agent.sample(state)

                    next_state, reward, done = env.step(action)

                    if done:
                        if memory.size() >= batch_size:
                            # 差分进化算法
                            de = DE(population, memory.sample(batch_size), size=time_step)
                            best_policy = de.evolution()
                            # sac_agent.alg.hard_update_target(best_policy)
                            population = []
                        break

                    episode_reward += reward
                    memory.append(state, action, reward, next_state)
                    state = next_state

                    population.append(sac_agent.alg.get_actor())
                    # 收集到足够的经验后进行网络的更新
                    if memory.size() >= warmup_steps:
                        # 梯度更新
                        batch_state, batch_action, batch_reward, batch_next_state = memory.sample(batch_size)
                        critic_loss, actor_loss = sac_agent.learn(batch_state, batch_action, batch_reward,
                                                                  batch_next_state)
                        writer.add_scalar('critic loss', critic_loss.numpy(), learn_steps)
                        writer.add_scalar('actor loss', actor_loss.numpy(), learn_steps)

                if max_episode_reward < episode_reward:
                    max_episode_reward = episode_reward
                    sac_agent.alg.save()

                episode_reward_list.append(episode_reward)

                writer.add_scalar('episode reward', episode_reward, len(episode_reward_list))

                if (epoch + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (epochs / 10 * i + epoch + 1),
                                      'episode_reward_list': '%.3f' % np.mean(episode_reward_list[-10:])})
                pbar.update(1)

    return episode_reward_list


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    epochs = 2500

    # 初始化超参数
    WARMUP_STEPS = 5000
    MEMORY_SIZE = int(1e6)
    BATCH_SIZE = 256
    GAMMA = 0.99
    TAU = 0.001
    ACTOR_LR = 3e-4
    CRITIC_LR = 3e-4
    ALPHA = 0.1

    # 定义环境、实例化模型
    env = Environment()
    state_dim = 30
    action_dim = 1

    # 初始化 模型，算法，智能体以及经验池
    model = ActorCriticModel(state_dim, action_dim)
    algorithm = SAC(model, gamma=GAMMA, tau=TAU, alpha=ALPHA, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    sac_agent = SACAgent(algorithm)
    memory = Memory(max_size=MEMORY_SIZE, state_dim=state_dim, action_dim=action_dim)

    # 开始训练
    train_begin_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = './' + train_begin_time + '/log'
    model_dir = './' + train_begin_time + '/model'
    writer = LogWriter(logdir=log_dir)
    print('visualdl --logdir ' + log_dir + ' --port 8080')
    episode_reward_list = train_off_policy_agent(env, action_dim, sac_agent, epochs, memory, WARMUP_STEPS, BATCH_SIZE)

    # 可视化训练结果
    episodes_list = list(range(len(episode_reward_list)))
    mv_return = moving_average(episode_reward_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SAC on {}'.format('iLab'))
    plt.show()
