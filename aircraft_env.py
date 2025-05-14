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
    def __init__(self, map_size=(100, 100), num_agents=5, num_obstacles=10, num_targets=8):
        super().__init__()
        self.map_size = map_size
        self.num_agents = num_agents
        self.num_obstacles = int(num_obstacles * map_size[0] / 50)
        self.num_targets = num_targets

        self.obstacle_buffer = 2  # 障碍物之间的安全距离
        self.agent_spawn_buffer = 6  # 飞机生成与障碍物的安全距离
        self.target_spawn_buffer = 3 # 目标点生成与障碍物的安全距离

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

        for _ in range(self.num_targets):
            attempt = 0
            while attempt < max_attempts:
                x = random.randint(3, self.map_size[0]-3)
                y = random.randint(3, self.map_size[1]-3)

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
                    if dist < 20:  # 目标点之间的最小距离
                        too_close = True
                        break

                if not too_close:
                    self.targets.append({
                        'x': x,
                        'y': y,
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
            y = self.map_size[1]
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
                'theta': -math.pi/2,
                'F': 0,
                'dead': False,
                'die_time': 0,
                'vision': 120,
                'volume': 50,
                'bulleted_num': 0,
                'damaged': 0,
                'blood': 100,
                'healthy': 100,
                'attack_range': 5
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

    def _init_sth(self):
        # 初始化飞机位置
        self.agents = []
        for _ in range(self.num_agents):
            agent = {
                'z': 1,
                'x': random.randint(1, self.map_size[0]),
                'y': random.randint(1, self.map_size[1]),
                'speed_x': random.uniform(-5 / 60, 5 / 60),  # 假设FPS=60
                'speed_y': random.uniform(15 / 60, 20 / 60),
                'speed_z': 0,
                'theta': random.uniform(0, 2 * math.pi),
                'F': 0,
                'dead': False,
                'die_time': 0,
                'vision': 120,
                'volume': 50,
                'bulleted_num': 0,
                'damaged': 0,
                'blood': 100,
                'healthy': 100,
                'attack_range': 5  # 新增攻击范围属性
            }
            self.agents.append(agent)

        # 初始化障碍物
        self.obstacles = []
        for _ in range(self.num_obstacles):
            obstacle = {
                'x': random.randint(1, self.map_size[0] - 1),
                'y': random.randint(1, self.map_size[1] - 1),
                'radius': 3
            }
            self.obstacles.append(obstacle)

        # 初始化目标点
        self.targets = []
        for _ in range(self.num_targets):
            target = {
                'x': random.randint(1, self.map_size[0] - 1),
                'y': random.randint(1, self.map_size[1] - 1),
                'active': True
            }
            self.targets.append(target)
    def step(self, actions):
        rewards = np.zeros(self.num_agents)
        dones = np.full(self.num_agents, False)
        truncated = np.full(self.num_agents, False)
        info = {'attacks': []}

        # 处理每个智能体的动作
        for i, agent in enumerate(self.agents):
            if agent['dead']:
                continue

            # 解析动作 (航向和油门)
            heading_action, throttle_action = actions[i]

            # 更新航向 ([-1,1] 映射到 [-0.1,0.1] 弧度变化)
            agent['theta'] += heading_action * 0.1
            agent['theta'] = agent['theta'] % (2 * math.pi)

            # 更新油门 ([-1,1] 映射到 [0,1] 并应用到最大速度)
            max_speed = 20 / 60  # 最大速度
            speed = (throttle_action + 1) / 2 * max_speed

            # 更新速度
            agent['speed_x'] = speed * math.cos(agent['theta'])
            agent['speed_y'] = speed * math.sin(agent['theta'])

            # 更新位置
            agent['x'] += agent['speed_x']
            agent['y'] += agent['speed_y']

            # 边界检查
            agent['x'] = max(0, min(self.map_size[0], agent['x']))
            agent['y'] = max(0, min(self.map_size[1], agent['y']))

            # 检查与障碍物的碰撞
            for obstacle in self.obstacles:
                dist = math.sqrt((agent['x'] - obstacle['x']) ** 2 + (agent['y'] - obstacle['y']) ** 2)
                if dist < obstacle['radius']:
                    # 碰撞惩罚
                    rewards[i] -= 10
                    agent['damaged'] += 20
                    agent['blood'] -= 20

                    # 反弹效果
                    angle = math.atan2(agent['y'] - obstacle['y'], agent['x'] - obstacle['x'])
                    agent['x'] = obstacle['x'] + math.cos(angle) * (obstacle['radius']) * 2
                    agent['y'] = obstacle['y'] + math.sin(angle) * (obstacle['radius']) * 2

            # 检查与目标点的距离
            for j, target in enumerate(self.targets):
                if not target['active']:
                    continue

                dist = math.sqrt((agent['x'] - target['x']) ** 2 + (agent['y'] - target['y']) ** 2)

                # 检查是否在攻击范围内
                if dist <= agent['attack_range'] and agent['volume'] > 0:
                    # 攻击目标
                    agent['volume'] -= 1
                    info['attacks'].append((i, j))

                    # 击中目标奖励
                    rewards[i] += 5

                    # 记录被攻击的目标
                    if j not in self.attacked_targets:
                        self.attacked_targets.append(j)
                        target['active'] = False

                    # 击中其他飞机的额外奖励
                    for k, other_agent in enumerate(self.agents):
                        if k != i and not other_agent['dead']:
                            other_dist = math.sqrt((other_agent['x'] - target['x']) ** 2 +
                                                   (other_agent['y'] - target['y']) ** 2)
                            if other_dist < 30:  # 假设目标周围30单位范围内的飞机都会被击中
                                other_agent['damaged'] += 25
                                other_agent['blood'] -= 25
                                other_agent['bulleted_num'] += 1
                                rewards[i] += 10  # 击中其他飞机的额外奖励

            # 检查生命值
            if agent['blood'] <= 0:
                agent['dead'] = True
                agent['die_time'] = 0
                rewards[i] -= 20  # 死亡惩罚

            # 简单奖励机制：靠近未被攻击的目标点奖励
            min_dist = float('inf')
            for target in self.targets:
                if target['active']:
                    dist = math.sqrt((agent['x'] - target['x']) ** 2 + (agent['y'] - target['y']) ** 2)
                    min_dist = min(min_dist, dist)

            if min_dist != float('inf'):
                rewards[i] += -min_dist * 0.01  # 距离越近奖励越高

        # 检查是否所有目标都被攻击
        all_targets_attacked = all(not target['active'] for target in self.targets)
        if all_targets_attacked:
            dones[:] = True
            rewards += 100  # 完成所有目标的额外奖励

        # 检查是否所有飞机都死亡
        all_agents_dead = all(agent['dead'] for agent in self.agents)
        if all_agents_dead:
            dones[:] = True

        self.agent_positions = self._get_position()
        return self._get_observations(), rewards, dones, truncated, info

    def _get_observations(self):
        observations = []
        for agent in self.agents:
            obs = np.array([
                agent['x'] / self.map_size[0],
                agent['y'] / self.map_size[1],
                agent['z'] / 1,
                agent['theta'] / 2 / math.pi,
                math.sqrt(agent['speed_x'] ** 2 + agent['speed_y'] ** 2),
                agent['volume'] / 50,
                agent['blood'] / 100
            ])
            observations.append(obs)
        return np.array(observations)

    def _get_position(self):
        return [(agent["x"], agent["y"]) for agent in self.agents]

    def render(self, mode='human'):
        # 渲染方法将在Qt界面中实现
        pass    