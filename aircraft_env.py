import numpy as np
from gym import Env, spaces
import random
import math

def is_overlap(x1, y1, r1, x2, y2, r2):
    """判断两个圆是否重叠"""
    return (x1 - x2) ** 2 + (y1 - y2) ** 2 < (r1 + r2) ** 2

def is_point_in_circle(px, py, cx, cy, cr):
    """判断点是否在圆内"""
    return (px - cx) ** 2 + (py - cy) ** 2 < cr ** 2

class AircraftEnv(Env):
    def __init__(self, map_size=(750, 650), num_agents=5, num_obstacles=0, num_targets=8):
        super().__init__()
        self.map_size = map_size
        self.num_agents = num_agents
        self.num_obstacles = num_obstacles
        self.num_targets = math.ceil(num_agents / 3)
        self.obstacle_buffer = 2  # 障碍物之间的安全距离
        self.agent_spawn_buffer = 6  # 飞机生成与障碍物的安全距离
        self.target_spawn_buffer = 3 # 目标点生成与障碍物的安全距离
        self.dt = 0.1
        self.speed = 10
        self._generate_obstacles()
        self._generate_targets()
        self._initialize_agents()
        self.num_obstacles = len(self.obstacles)
        self.num_targets = len(self.targets)

        self.max_steps = 1000

        # 定义动作空间 (航向和油门)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # 定义状态空间 (x, y, z坐标, 航向, 速度, 剩余弹药, 生命值)
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(7,),
            dtype=np.float32
        )

        # 初始化飞机位置和环境

        self.agent_num = num_agents
        self.obs_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        self.reset()

    def _generate_obstacles(self):
        self.obstacles = []
        obstacle_radius = 3
        self.max_attempts = 1000
        for _ in range(self.num_obstacles):
            for _ in range(self.max_attempts):
                x = random.randint(obstacle_radius, self.map_size[0] - obstacle_radius)
                y = random.randint(obstacle_radius, self.map_size[1] - 20)
                # 检查与已有障碍物不重叠
                if all(not is_overlap(x, y, obstacle_radius + self.target_spawn_buffer, o['x'], o['y'], o['radius']) for o in self.obstacles):
                    self.obstacles.append({'x': x, 'y': y, 'radius': obstacle_radius})
                    break
            else:
                print("障碍物初始化失败，尝试次数过多")

    def _generate_targets(self):
        self.targets = []
        max_attempts = 1000

        for i in range(self.num_targets):
            attempt = 0
            while attempt < max_attempts:
                x = random.randint(150, self.map_size[0]-150)
                y = random.randint(150, self.map_size[1]-400)

                # 检查与障碍物的距离
                too_close = False
                for obs in self.obstacles:
                    dist = math.sqrt((x - obs['x']) ** 2 + (y - obs['y']) ** 2)
                    if dist < (obs['radius'] + self.target_spawn_buffer):
                        too_close = True
                        break

                # 检查与其他目标点的距离
                for target in self.targets:
                    dist = math.sqrt((x - target['x']) ** 2 + (y - target['y']) ** 2)
                    if dist < 100:  # 目标点之间的最小距离
                        too_close = True
                        break

                if not too_close:
                    self.targets.append({
                        'id': i,
                        'x': x,
                        'y': y,
                        'speed_x': random.uniform(-5 / 60, 5 / 60),
                        'speed_y': random.uniform(15 / 60, 20 / 60),
                        'speed_z': 0,
                        'speed': self.speed,
                        'theta': math.pi/2,
                        'blood': 100,
                        'active': True
                    })
                    break

                attempt += 1

    def _initialize_agents(self):
        self.agents = []
        positions = []

        # 下边
        for i in range(self.num_agents):
            x = int((i + 1) * self.map_size[0] / (self.num_agents+1))
            y = self.map_size[1] - 50
            positions.append((x, y))

        for x, y in positions:
            # 检查不与障碍物/目标点重叠
            if any(is_overlap(x, y, 1, o['x'], o['y'], o['radius']) for o in self.obstacles):
                continue
            if any(x == t['x'] and y == t['y'] for t in self.targets):
                continue
            agent = {
                'z': 1,
                'x': x,
                'y': y,
                'speed_x': random.uniform(-5 / 60, 5 / 60),
                'speed_y': random.uniform(15 / 60, 20 / 60),
                'speed_z': 0,
                'speed': self.speed * 2,
                'theta': random.uniform(0,2*math.pi),
                'F': 0,
                'dead': False,
                'die_time': 0,
                'vision': 120,
                'volume': 5,
                'bulleted_num': 40,
                'damaged': 0,
                'blood': 100,
                'healthy': 100,
                'attack_range': 30
            }
            self.agents.append(agent)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._generate_obstacles()
        self._generate_targets()
        self._initialize_agents()

        # 初始化被攻击的目标
        self.attacked_targets = []
        self.agent_positions = self._get_position()
        return self._get_observations(), {}


    def step(self, actions):
        rewards = np.zeros(self.num_agents)
        dones = np.full(self.num_agents, False)
        truncated = np.full(self.num_agents, False)
        info = {'attacks': []}

        # 新增：时间惩罚系数
        TIME_PENALTY = 0.1
        # 新增：存活奖励系数
        SURVIVE_REWARD = 0.3
        # 新增：路径效率系数
        DIRECTION_REWARD_SCALE = 0.6
        for i, target in enumerate(self.targets):
            if not target['active']:
                continue
            heading_action = random.uniform(-5, 5)
            target['theta'] = target['theta']+0.6*heading_action*self.dt
            if target['theta'] > 2 * math.pi:
                target['theta'] = target['theta'] - 2 * math.pi
            elif target['theta'] < 0:
                target['theta'] = target['theta'] + 2 * math.pi
            # 更新速度
            target['speed_x'] = target['speed']*math.cos(target['theta'])*self.dt
            target['speed_y'] = target['speed']*math.sin(target['theta'])*self.dt
            # 更新位置
            target['x'] += target['speed_x']
            target['y'] -= target['speed_y']
            # 边界检查
            target['x'] = max(0, min(self.map_size[0], target['x']))
            target['y'] = max(0, min(self.map_size[1], target['y']))

        for i, agent in enumerate(self.agents):
            if agent['dead']:
                continue
            if not self.targets[i//3]['active']:
                dones[i] = True
            # 解析动作 (航向和油门)
            throttle_action, heading_action = actions[i]
            # 更新航向 ([-1,1] 映射到 [-0.1,0.1] 弧度变化)
            agent['theta'] = agent['theta']+0.6*heading_action*self.dt
            if agent['theta'] > 2 * math.pi:
                agent['theta'] = agent['theta'] - 2 * math.pi
            elif agent['theta'] < 0:
                agent['theta'] = agent['theta'] + 2 * math.pi

            # 更新油门 ([-1,1] 映射到 [0,1] 并应用到最大速度)
            agent['speed'] = agent['speed']+0.3*throttle_action*self.dt
            agent['speed'] = np.clip(agent['speed'], self.speed*2-10, self.speed*2)

            # 更新速度
            agent['speed_x'] = self.speed*math.cos(agent['theta'])*self.dt
            agent['speed_y'] = self.speed*math.sin(agent['theta'])*self.dt

            # 更新位置
            agent['x'] += agent['speed_x']
            agent['y'] -= agent['speed_y']

            # 边界检查
            agent['x'] = max(0, min(self.map_size[0], agent['x']))
            agent['y'] = max(0, min(self.map_size[1], agent['y']))

            dis = math.hypot(agent['x']-self.targets[int(i/3)]['x'],
                                    agent['y']-self.targets[int(i/3)]['y'])
            if dis < 40 and agent['bulleted_num'] > 0:
                self.targets[int(i/3)]['blood'] -= 1
                agent['bulleted_num'] -= 1
                if self.targets[int(i / 3)]['blood'] <= 0:
                    self.targets[int(i/3)]['active'] = False
            rewards[i] -= 0.05 * dis
            rewards[i] += 0 if dis > 50 else 2.5
        # print(f"rewards: {rewards}")
        self.agent_positions = self._get_position()
        return self._get_observations(), rewards, dones, truncated, info

    def _get_observations(self):
        observations = []
        for idx, agent in enumerate(self.agents):
            # 自身状态信息
            self_info = np.array([
                agent['x'] / 1000,
                agent['y'] / 1000,
                agent['speed'] / 30,
                agent['theta'] * 57.3 / 360,
            ])

            # 障碍物信息
            obstacle_info = []
            for obstacle in self.obstacles:
                dist = math.sqrt((agent['x'] - obstacle['x']) ** 2 + (agent['y'] - obstacle['y']) ** 2)
                angle = math.atan2(obstacle['y'] - agent['y'], obstacle['x'] - agent['x'])
                obstacle_info.extend([dist / max(self.map_size), angle / (2 * math.pi)])

            # 目标点信息
            target_info = []
            target = self.targets[idx//3]
            target_info.extend([target['x']/ 1000, target['y']/ 1000, target['speed'] / 30])

            obs = np.concatenate([self_info, obstacle_info, target_info])
            observations.append(obs)

        return np.array(observations)

    def _get_position(self):
        return [(agent["x"], agent["y"]) for agent in self.agents]

    def render(self, mode='human'):
        # 渲染方法将在Qt界面中实现
        pass    