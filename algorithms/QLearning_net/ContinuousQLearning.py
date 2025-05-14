import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ContinuousQLearning:
    def __init__(self, env, lr=0.001, gamma=0.99, epsilon=0.1, **kwargs):
        self.env = env
        self.num_agents = env.num_agents
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        # 为每个智能体创建Q网络和优化器
        self.q_networks = [QNetwork(self.state_dim, self.action_dim) for _ in range(self.num_agents)]
        self.optimizers = [optim.Adam(q_net.parameters(), lr=lr) for q_net in self.q_networks]
        self.criterion = nn.MSELoss()

    def get_action(self, state, agent_id):
        # epsilon-贪心策略
        if np.random.random() < self.epsilon:
            # 随机动作（探索）
            return self.env.action_space.sample()
        else:
            # 利用Q网络选择最佳动作
            state_tensor = torch.FloatTensor(state[agent_id])
            with torch.no_grad():
                q_values = self.q_networks[agent_id](state_tensor)
                # 直接将Q值作为动作输出
                action = q_values.numpy()
                # 将动作裁剪到有效范围内
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
                return action

    def update(self, state, actions, rewards, next_state, done):
        for i in range(self.num_agents):
            state_tensor = torch.FloatTensor(state[i])
            action_tensor = torch.FloatTensor(actions[i])
            reward_tensor = torch.FloatTensor([rewards[i]])
            next_state_tensor = torch.FloatTensor(next_state[i])
            done_tensor = torch.FloatTensor([done])

            # 计算目标Q值
            with torch.no_grad():
                next_q_values = self.q_networks[i](next_state_tensor)
                max_next_q_value = torch.max(next_q_values)
                td_target = reward_tensor + self.gamma * max_next_q_value * (1 - done_tensor)

            # 计算当前Q值
            current_q_value = self.q_networks[i](state_tensor)
            # 计算动作对应的Q值（这里简化处理，使用点积）
            current_q_value = torch.sum(current_q_value * action_tensor)

            # 计算损失并更新网络
            loss = self.criterion(current_q_value.unsqueeze(0), td_target)
            self.optimizers[i].zero_grad()
            loss.backward()
            self.optimizers[i].step()