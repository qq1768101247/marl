import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import numpy as np
from net.net import Actor, Critic, Memory, Ornstein_Uhlenbeck_Noise, Entroy


class MASACAlgorithm:
    def __init__(self, env,
                 gamma=0.9, batch_size=2000, memory_capacity=20000):
        # 初始化参数
        self.env = env
        self.n_agents = env.num_agents
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.action_dim = self.action_space.shape[0]
        self.state_dim = self.observation_space.shape[0]

        self.actors = [Actor(self.state_dim, self.action_dim) for _ in range(self.n_agents)]
        self.critics = [Critic(self.state_dim, self.state_dim, self.n_agents, self.n_agents, self.action_dim)
                        for _ in range(self.n_agents)]

        # 熵调节参数
        self.entroys = [Entroy() for _ in range(self.n_agents)]

        # 经验回放缓冲区
        total_obs_dim = self.n_agents * self.state_dim
        total_action_dim = self.n_agents * self.action_dim
        self.memory = Memory(memory_capacity, total_obs_dim + total_action_dim + total_obs_dim + self.n_agents)

        # 噪声生成器
        self.ou_noise = Ornstein_Uhlenbeck_Noise(mu=np.zeros((self.n_agents, self.action_dim)))
        self.timestep = 0

    def flatten_obs(self, obs_list):
        # 将所有智能体的观测展平为一维数组
        return np.concatenate([o.flatten() for o in obs_list])

    def get_action(self, agent_id, state):
        # 获取单个智能体的动作
        action = self.actors[agent_id].choose_action(state[agent_id])

        # 添加探索噪声（训练初期使用）
        if self.timestep <= 2000:  # 前2000步添加噪声
            noise = self.ou_noise()[agent_id]
            action = action + noise

        action = np.clip(action, self.action_space.low[0], self.action_space.high[0])
        self.timestep += 1
        return action

    def update(self, state, actions, rewards, next_state, dones):
        # 存储经验到回放缓冲区
        flat_state = self.flatten_obs(state)
        flat_actions = np.array(actions).flatten()
        flat_rewards = np.array(rewards).flatten()
        flat_next_state = self.flatten_obs(next_state)

        self.memory.store_transition(flat_state, flat_actions, flat_rewards, flat_next_state)

        # 当缓冲区中有足够数据时进行学习
        if self.memory.memory_counter >= self.batch_size:
            self._learn()

    def _learn(self):
        # 从经验回放中采样
        b_M = self.memory.sample(self.batch_size)

        total_obs_dim = self.n_agents * self.state_dim
        total_action_dim = self.n_agents * self.action_dim

        # 解析样本
        b_s = b_M[:, :total_obs_dim]
        b_a = b_M[:, total_obs_dim: total_obs_dim + total_action_dim]
        b_r = b_M[:, -total_obs_dim - self.n_agents: -total_obs_dim]
        b_s_ = b_M[:, -total_obs_dim:]

        # 更新所有智能体网络
        for i in range(self.n_agents):
            # 计算目标Q值
            new_action, log_prob_ = self.actors[i].evaluate(
                b_s_[:, self.state_dim * i:self.state_dim * (i + 1)])
            target_q1, target_q2 = self.critics[i].target_critic_v(b_s_, new_action)
            target_q = b_r[:, i:i + 1] + self.gamma * (
                    torch.min(target_q1, target_q2) - self.entroys[i].alpha * log_prob_)

            # 更新评论家网络
            current_q1, current_q2 = self.critics[i].get_v(b_s, b_a[:,
                                                                self.action_dim * i:self.action_dim * (i + 1)])
            self.critics[i].learn(current_q1, current_q2, target_q.detach())

            # 更新行动者网络
            a, log_prob = self.actors[i].evaluate(
                b_s[:, self.state_dim * i:self.state_dim * (i + 1)])
            q1, q2 = self.critics[i].get_v(b_s, a)
            q = torch.min(q1, q2)
            actor_loss = (self.entroys[i].alpha * log_prob - q).mean()

            # 更新熵调节参数
            alpha_loss = -(self.entroys[i].log_alpha.exp() * (
                    log_prob + self.entroys[i].target_entropy).detach()).mean()

            self.actors[i].learn(actor_loss)
            self.entroys[i].learn(alpha_loss)
            self.entroys[i].alpha = self.entroys[i].log_alpha.exp()

            # 软更新目标网络
            self.critics[i].soft_update()

    def save_models(self, path="models"):
        # 保存所有模型参数
        for i in range(self.n_agents):
            save_data = {'net': self.actors[i].actor_net.state_dict(),
                         'opt': self.actors[i].optimizer.state_dict()}
            torch.save(save_data, f"{path}/actor_{i}.pth")