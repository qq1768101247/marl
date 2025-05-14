import numpy as np

# 强化学习算法基类
class RLAlgorithm:
    def __init__(self, env):
        self.env = env
        
    def train(self, num_episodes=100):
        raise NotImplementedError("子类必须实现此方法")
    
    def get_action(self, state):
        raise NotImplementedError("子类必须实现此方法")

# Q-learning算法实现
class QLearning(RLAlgorithm):
    def __init__(self, env):
        super().__init__(env)
        self.alpha = 0.1  # 学习率
        self.gamma = 0.9  # 折扣因子
        self.epsilon = 0.1  # 探索率

        self.num_agents = env.num_agents
        # 离散化动作空间
        self.num_actions_per_dim = 8  # 每个维度的离散动作数量
        self.action_space = env.action_space
        # 为每个智能体创建离散动作列表
        self.discrete_actions = []
        for i in range(self.num_agents):
            # 生成每个维度的离散值
            action_ranges = [
                np.linspace(
                    self.action_space.low[j],
                    self.action_space.high[j],
                    self.num_actions_per_dim
                )
                for j in range(self.action_space.shape[0])
            ]

            # 生成所有可能的动作组合
            actions = np.array(np.meshgrid(*action_ranges)).T.reshape(-1, self.action_space.shape[0])
            self.discrete_actions.append(actions)
        
        # 初始化Q表
        self.q_tables = [{} for _ in range(env.num_agents)]
        
    def get_action(self, agent_idx, state):
        state_tuple = tuple(state[agent_idx])
        # 探索 vs 利用
        if np.random.uniform(0, 1) < self.epsilon:
            # 随机动作（探索）
            action_index = np.random.randint(0, len(self.discrete_actions[agent_idx]))
        else:
            # 利用当前Q表选择最佳动作
            if state_tuple not in self.q_tables[agent_idx]:
                self.q_tables[agent_idx][state_tuple] = np.zeros(len(self.discrete_actions[agent_idx]))

            action_values = self.q_tables[agent_idx][state_tuple]
            action_index = np.argmax(action_values)

        # 返回实际的连续动作值
        return self.discrete_actions[agent_idx][action_index]

    def update(self, state, actions, rewards, next_state, done):
        # 更新Q表
        for i in range(self.num_agents):
            state_tuple = tuple(state[i])
            next_state_tuple = tuple(next_state[i])

            # 找到执行的动作在离散动作列表中的索引
            action = actions[i]
            action_index = np.argmin(np.linalg.norm(self.discrete_actions[i] - action, axis=1))

            # 初始化Q值（如果状态未见过）
            if state_tuple not in self.q_tables[i]:
                self.q_tables[i][state_tuple] = np.zeros(len(self.discrete_actions[i]))
            if next_state_tuple not in self.q_tables[i]:
                self.q_tables[i][next_state_tuple] = np.zeros(len(self.discrete_actions[i]))

            # Q-learning更新公式
            td_target = rewards[i] + self.gamma * np.max(self.q_tables[i][next_state_tuple])
            td_error = td_target - self.q_tables[i][state_tuple][action_index]
            self.q_tables[i][state_tuple][action_index] += self.alpha * td_error

    def train(self, num_episodes=100):
        rewards_history = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = np.zeros(self.env.num_agents)
            
            done = False
            while not done:
                # 获取所有智能体的动作
                actions = [self.get_action(i, state) for i in range(self.env.num_agents)]
                
                # 执行动作
                next_state, rewards, dones, truncated, _ = self.env.step(actions)
                episode_reward += rewards
                
                # 更新Q表
                for i in range(self.env.num_agents):
                    state_tuple = tuple(state[i])
                    next_state_tuple = tuple(next_state[i])
                    action = actions[i]
                    
                    # 初始化Q值（如果状态未见过）
                    if state_tuple not in self.q_tables[i]:
                        self.q_tables[i][state_tuple] = np.zeros(self.env.action_space.n)
                    if next_state_tuple not in self.q_tables[i]:
                        self.q_tables[i][next_state_tuple] = np.zeros(self.env.action_space.n)
                    
                    # Q-learning更新公式
                    td_target = rewards[i] + self.gamma * np.max(self.q_tables[i][next_state_tuple])
                    td_error = td_target - self.q_tables[i][state_tuple][action]
                    self.q_tables[i][state_tuple][action] += self.alpha * td_error
                
                state = next_state
                done = all(dones) or all(truncated)
            
            rewards_history.append(episode_reward)
            print(f"Episode {episode+1}/{num_episodes}, Reward: {np.mean(episode_reward)}")
        
        return rewards_history

# PPO算法实现（简化版）
class PPO(RLAlgorithm):
    def __init__(self, env):
        super().__init__(env)
        # 简化实现，实际需要神经网络
        self.policy_networks = [{"weights": np.random.rand(env.observation_space.shape[0], env.action_space.shape[0])}
                               for _ in range(env.num_agents)]
        self.value_networks = [{"weights": np.random.rand(env.observation_space.shape[0], 1)} 
                              for _ in range(env.num_agents)]
        
    def get_action(self, agent_idx, state):
        # 简化的策略网络前向传播
        state = state[agent_idx].reshape(-1)
        mean = np.dot(state, self.policy_networks[agent_idx]["weights"])
        action = np.random.normal(mean, [0.2,0.2])
        # 确保动作在Box空间的范围内
        low, high = self.env.action_space.low, self.env.action_space.high
        action = np.clip(action, low, high)
        return action

    def update(self, state, actions, rewards, next_state, done):
        pass
