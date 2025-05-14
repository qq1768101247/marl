import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import List, Dict, Tuple, Optional
from mappo.utils.replay_buffer import ReplayBuffer
from mappo.networks.mlp import MLPNetwork  # 添加了缺失的导入

class RLAlgorithm:
    def __init__(self, env):
        self.env = env

class MAPPO(RLAlgorithm):
    def __init__(self, 
                 env, 
                 lr=1e-3, 
                 gamma=0.99, 
                 clip_epsilon=0.2, 
                 ent_coef=0.01, 
                 vf_coef=0.5,
                 gae_lambda=0.95,
                 max_grad_norm=0.5,
                 batch_size=64,
                 buffer_size=10000,
                 hidden_dims=[64, 64],
                 activation=nn.Tanh,
                 epochs=10):
        super().__init__(env)
        self.lr = lr
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.epochs = epochs
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化策略网络和价值网络
        self.policy_networks = nn.ModuleList([
            MLPNetwork(
                env.observation_space.shape[0], 
                env.action_space.shape[0],
                hidden_dims=hidden_dims,
                activation=activation
            ).to(self.device)
            for _ in range(env.num_agents)
        ])
        
        self.value_networks = nn.ModuleList([
            MLPNetwork(
                env.observation_space.shape[0], 
                1,
                hidden_dims=hidden_dims,
                activation=activation
            ).to(self.device)
            for _ in range(env.num_agents)
        ])
        
        # 策略标准差，可训练参数
        self.log_stds = nn.ParameterList([
            nn.Parameter(torch.zeros(env.action_space.shape[0]))
            for _ in range(env.num_agents)
        ])
        
        # 优化器
        self.policy_optimizers = [
            optim.Adam(
                list(policy.parameters()) + [log_std],
                lr=lr
            )
            for policy, log_std in zip(self.policy_networks, self.log_stds)
        ]
        
        self.value_optimizers = [
            optim.Adam(value.parameters(), lr=lr)
            for value in self.value_networks
        ]
        
        # 经验回放缓冲区
        self.replay_buffers = [
            ReplayBuffer(buffer_size=buffer_size)
            for _ in range(env.num_agents)
        ]
        
    def get_action(self, agent_idx, state, deterministic=False):
        state = torch.FloatTensor(state[agent_idx]).to(self.device).unsqueeze(0)
        policy = self.policy_networks[agent_idx]
        log_std = self.log_stds[agent_idx]
        
        # 策略网络前向传播
        mean = policy(state)
        std = torch.exp(log_std)
        
        # 创建分布
        dist = Normal(mean, std)
        
        if deterministic:
            action = mean.detach().cpu().numpy()[0]
        else:
            action = dist.sample().detach().cpu().numpy()[0]
        
        # 确保动作在Box空间范围内
        low, high = self.env.action_space.low, self.env.action_space.high
        action = np.clip(action, low, high)
        
        # 计算动作对数概率
        log_prob = dist.log_prob(torch.FloatTensor(action).to(self.device)).sum().item()
        
        # 计算状态价值
        value = self.value_networks[agent_idx](state).item()

        self.log_prob = log_prob
        self.value = value
        return action
    
    def store_transition(self, state, action, log_prob, value, reward, done):

        for agent_idx in range(self.env.num_agents):
            self.replay_buffers[agent_idx].add(
                state=state[agent_idx],
                action=action,
                log_prob=log_prob,
                value=value,
                reward=reward,
                done=done
            )
    
    def compute_returns_and_advantages(self, agent_idx, next_value, dones):
        buffer = self.replay_buffers[agent_idx]
        rewards = buffer.rewards
        values = buffer.values
        
        returns = []
        advantages = []
        R = next_value
        last_gae_lam = 0
        
        for i in reversed(range(len(rewards))):
            done = dones[i] if i < len(dones) else dones[-1]
            reward = rewards[i]
            value = values[i]
            
            # 计算TD误差
            delta = reward + self.gamma * R * (1 - done) - value
            
            # GAE计算
            last_gae_lam = delta + self.gamma * self.gae_lambda * last_gae_lam * (1 - done)
            advantages.append(last_gae_lam)
            
            R = reward + self.gamma * R * (1 - done)
            returns.append(R)
        
        # 反转列表使时间顺序正确
        advantages = advantages[::-1]
        returns = returns[::-1]
        
        # 标准化优势函数
        advantages = np.array(advantages)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        return returns, advantages
    
    def update(self, state, actions, rewards, next_state, dones):

        # 对每个智能体进行更新
        for agent_idx in range(self.env.num_agents):
            buffer = self.replay_buffers[agent_idx]
            if len(buffer) < self.batch_size:
                continue
                
            # 获取最后一个状态的值作为引导
            last_state = torch.FloatTensor(buffer.states[-1]).to(self.device).unsqueeze(0)
            next_value = self.value_networks[agent_idx](last_state).item()
            
            # 计算回报和优势
            dones = buffer.dones
            returns, advantages = self.compute_returns_and_advantages(agent_idx, next_value, dones)
            
            # 转换为张量
            states = torch.FloatTensor(buffer.states).to(self.device)
            actions = torch.FloatTensor(buffer.actions).to(self.device)
            old_log_probs = torch.FloatTensor(buffer.log_probs).to(self.device)
            returns = torch.FloatTensor(returns).to(self.device)
            advantages = torch.FloatTensor(advantages).to(self.device)
            
            # 创建数据集和数据加载器
            dataset = torch.utils.data.TensorDataset(
                states, actions, old_log_probs, returns, advantages
            )
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True
            )
            
            # 更新策略网络和价值网络
            policy = self.policy_networks[agent_idx]
            value = self.value_networks[agent_idx]
            log_std = self.log_stds[agent_idx]
            policy_optimizer = self.policy_optimizers[agent_idx]
            value_optimizer = self.value_optimizers[agent_idx]
            
            for _ in range(self.epochs):
                for batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages in dataloader:
                    # 策略更新
                    policy_optimizer.zero_grad()
                    
                    # 前向传播
                    means = policy(batch_states)
                    stds = torch.exp(log_std)
                    dist = Normal(means, stds)
                    
                    # 计算当前对数概率
                    log_probs = dist.log_prob(batch_actions).sum(dim=1)
                    
                    # 计算策略比率
                    ratio = torch.exp(log_probs - batch_old_log_probs)
                    
                    # 计算策略损失
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # 计算熵损失
                    entropy = dist.entropy().mean()
                    entropy_loss = -self.ent_coef * entropy
                    
                    # 总策略损失
                    policy_total_loss = policy_loss + entropy_loss
                    policy_total_loss.backward()
                    
                    # 梯度裁剪
                    nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(log_std, self.max_grad_norm)
                    
                    policy_optimizer.step()
                    
                    # 价值更新
                    value_optimizer.zero_grad()
                    
                    # 计算价值损失
                    values = value(batch_states).squeeze()
                    value_loss = self.vf_coef * torch.mean((values - batch_returns) ** 2)
                    
                    value_loss.backward()
                    
                    # 梯度裁剪
                    nn.utils.clip_grad_norm_(value.parameters(), self.max_grad_norm)
                    
                    value_optimizer.step()    