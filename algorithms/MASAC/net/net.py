# --------------------- 定义不同观测维度的智能体网络 ---------------------
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import constants as C

policy_lr = 1e-3
value_lr = 3e-3
tau = 1e-2

class ActorNet(nn.Module):
    """Actor，观测维度 state_dim"""
    def __init__(self,inp,outp):
        super(ActorNet, self).__init__()
        self.in_to_y1=nn.Linear(inp,256)
        self.in_to_y1.weight.data.normal_(0,0.1)
        self.y1_to_y2=nn.Linear(256,256)
        self.y1_to_y2.weight.data.normal_(0,0.1)
        self.out=nn.Linear(256,outp)
        self.out.weight.data.normal_(0,0.1)
        self.std_out = nn.Linear(256, outp)
        self.std_out.weight.data.normal_(0, 0.1)

    def forward(self,inputstate):
        inputstate=self.in_to_y1(inputstate)
        inputstate=F.relu(inputstate)
        inputstate=self.y1_to_y2(inputstate)
        inputstate=F.relu(inputstate)
        mean=C.max_action*torch.tanh(self.out(inputstate))#输出概率分布的均值mean
        log_std=self.std_out(inputstate)#softplus激活函数的值域>0
        log_std=torch.clamp(log_std,-20,2)
        std=log_std.exp()
        return mean,std


class Actor:
    def __init__(self, state_dim, action_number):
        self.actor_net=ActorNet(state_dim,action_number)#这只是均值mean
        self.optimizer=torch.optim.Adam(self.actor_net.parameters(),lr=policy_lr)

    def choose_action(self,s):
        inputstate = torch.as_tensor(s, device=C.device, dtype=C.DTYPE)
        mean,std=self.actor_net(inputstate)
        dist = torch.distributions.Normal(mean, std)
        action=dist.sample()
        action=torch.clamp(action,C.min_action,C.max_action)
        return action.cpu().detach().numpy()

    def evaluate(self, state):
        mean, log_std = self.actor_net(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # 重参数化采样
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

    def learn(self,actor_loss):
        loss=actor_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class CriticNet(nn.Module):
    """Critic，输入为全局状态+动作"""

    def __init__(self, state_dim_all, action_dim):
        super().__init__()
        # 共享特征提取层
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim_all + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        # 双Q网络头
        self.q1_head = nn.Linear(256, 1)
        self.q2_head = nn.Linear(256, 1)

        self._init_weights()
        self.to(device=C.device, dtype=C.DTYPE)

    def _init_weights(self):
        for layer in self.shared_net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        nn.init.uniform_(self.q1_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.q2_head.weight, -3e-3, 3e-3)

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=1)
        features = self.shared_net(sa)
        return self.q1_head(features), self.q2_head(features)  # 每个头输出单个Q值

class Critic:
    def __init__(self, state_dim_leader, N_Agent, action_dim):
        # self.critic_v,self.target_critic_v=(CriticNet(C.state_dim_leader * C.N_Agent + C.state_dim_follower * C.M_Enemy, C.action_number),
        #                                     CriticNet(C.state_dim_leader * C.N_Agent + C.state_dim_follower * C.M_Enemy, C.action_number))#改网络输入状态，生成一个Q值
        self.critic_v,self.target_critic_v=(CriticNet(state_dim_leader * N_Agent, action_dim),
                                            CriticNet(state_dim_leader * N_Agent, action_dim))#改网络输入状态，生成一个Q值
        self.target_critic_v.load_state_dict(self.critic_v.state_dict())
        self.optimizer = torch.optim.Adam(self.critic_v.parameters(), lr=value_lr,eps=1e-5)
        self.lossfunc = nn.MSELoss()
    def soft_update(self):
        for target_param, param in zip(self.target_critic_v.parameters(), self.critic_v.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def get_v(self,s,a):
        return self.critic_v(s,a)

    def target_get_v(self,s,a):
        return self.target_critic_v(s,a)

    def learn(self,current_q1,current_q2,target_q):
        loss = self.lossfunc(current_q1, target_q) + self.lossfunc(current_q2, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def compute_loss(self, current_q1, current_q2, target_q):
        return self.lossfunc(current_q1, target_q) + self.lossfunc(current_q2, target_q)


"""------------------------------------------------------------------"""

class Ornstein_Uhlenbeck_Noise:
    def __init__(self, mu, sigma=0.1, theta=0.1, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + \
            self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        '''
        后两行是dXt，其中后两行的前一行是θ(μ-Xt)dt，后一行是σεsqrt(dt)
        '''
        self.x_prev = x
        return x

    def reset(self):
        if self.x0 is not None:
            self.x_prev = self.x0
        else:
            self.x_prev = np.zeros_like(self.mu)

class Memory:
    def __init__(self, capacity, dims, device=C.device, dtype=C.DTYPE):
        self.capacity = capacity
        self.device = device
        self.dtype = dtype

        # 直接在 GPU 上初始化存储空间
        self.mem = torch.zeros((capacity, dims),
                               dtype=self.dtype,
                               device=self.device)

        self.memory_counter = 0

    def store_transition(self, s, a, r, s_):
        """存储转换经验到 GPU 内存"""
        # 将输入数据转换为 GPU Tensor
        s = torch.as_tensor(s, dtype=self.dtype, device=self.device)
        a = torch.as_tensor(a, dtype=self.dtype, device=self.device)
        r = torch.as_tensor(r, dtype=self.dtype, device=self.device)
        s_ = torch.as_tensor(s_, dtype=self.dtype, device=self.device)

        # 水平拼接并展平
        tran = torch.cat([s.flatten(),
                          a.flatten(),
                          r.flatten(),
                          s_.flatten()])

        index = self.memory_counter % self.capacity
        self.mem[index] = tran
        self.memory_counter += 1

    def sample(self, n):
        """从 GPU 内存中随机采样"""
        assert self.is_full, "Memory not enough"

        # 生成随机索引 (直接在 GPU 上操作)
        indices = torch.randint(0, self.capacity, (n,),
                                device=self.device)

        return self.mem[indices]  # 返回 GPU Tensor

    def delete(self, batch):
        self.memory_counter = self.memory_counter - batch
        self.mem = self.mem[batch:]


    @property
    def is_full(self):
        """检查记忆库是否已满"""
        return self.memory_counter >= self.capacity

q_lr = 3e-4
class Entroy:
    def __init__(self):
        self.target_entropy = -0.1
        self.log_alpha = torch.zeros(1, requires_grad=True, device=C.device, dtype=C.DTYPE)
        self.alpha = self.log_alpha.exp()
        self.optimizer = torch.optim.Adam([self.log_alpha], lr=q_lr)

    def learn(self,entroy_loss):
        loss=entroy_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()