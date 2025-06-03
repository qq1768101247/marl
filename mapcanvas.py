import math
import pickle
import time

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QImage, QPixmap
from PyQt5.QtWidgets import (QWidget, QVBoxLayout,
                             QLabel)
from matplotlib.ticker import MaxNLocator

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False


class MapCanvas(QWidget):
    def __init__(self, env, render=True, parent=None):
        super().__init__(parent)
        self.env = env
        self.setMinimumSize(400, 400)
        self.target_image = QImage("resource/image/enemy2.png")
        self.plane_image = QImage("resource/image/blue.png")
        self.bg_image = QImage("resource/image/bg.png")  # 你的背景图片路径
        self.bg_pixmap = None
        self.last_size = None
        self.render = render

        # 主更新计时器 - 控制UI刷新频率
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_canvas)
        self.update_timer.start(60 / 1000)  # 60fps

        # 加载爆炸动画帧
        self.explosion_frames = []
        for i in range(1, 7):  # 加载6帧爆炸图片
            frame = QImage(f"resource/image/enemy2_down{i}.png")
            self.explosion_frames.append(frame)

        # 爆炸效果管理：存储爆炸的位置和开始时间
        self.explosions = []
        # 是否已经播放过
        self.explosions_id = []

        # 爆炸动画帧率控制
        self.explosion_frame_rate = 100  # 每帧持续时间(毫秒)
        self.explosion_timer = QTimer(self)
        self.explosion_timer.timeout.connect(self.update_explosions)
        self.explosion_timer.start(self.explosion_frame_rate)


    def set_render(self, render):
        self.render = render

    def update_explosions(self):
        """更新爆炸动画状态"""
        current_time = time.time()
        # 过滤掉已经完成的爆炸动画
        self.explosions = [exp for exp in self.explosions
                           if current_time - exp['start_time'] < len(self.explosion_frames) * (
                                       self.explosion_frame_rate / 1000)]
        self.update()

    def check_for_new_explosions(self):
        """检查是否有新的目标变为不活跃状态，如果是则添加爆炸效果"""
        if not hasattr(self.env, 'targets'):
            return

        # 记录已经添加爆炸效果的目标ID
        existing_explosion_targets = [exp.get('target_id') for exp in self.explosions]

        for target in self.env.targets:
            if (not target['active'] and target['id'] not in existing_explosion_targets
                    and target['id'] not in self.explosions_id):
                self.explosions_id.append(target['id'])
                # 添加新爆炸效果
                rect = self.rect()
                margin = 20
                map_width = rect.width() - 2 * margin
                map_height = rect.height() - 2 * margin

                # 计算目标在画布上的位置
                x = margin + int(target['x'] * map_width / self.env.map_size[0])
                y = margin + int(target['y'] * map_height / self.env.map_size[1])

                self.explosions.append({
                    'x': x,
                    'y': y,
                    'start_time': time.time(),
                    'target_id': target['id']
                })

    def draw_explosions(self, painter, margin, map_width, map_height):
        """绘制所有爆炸动画"""
        current_time = time.time()

        for explosion in self.explosions:
            # 计算当前应该显示的帧
            elapsed_time = current_time - explosion['start_time']
            frame_index = int(elapsed_time / (self.explosion_frame_rate / 1000))

            if frame_index < len(self.explosion_frames):
                x = explosion['x']
                y = explosion['y']

                # 绘制爆炸帧
                frame = self.explosion_frames[frame_index]
                image_size = 40  # 爆炸图片大小，可以根据需要调整

                # 居中绘制爆炸图片
                painter.drawImage(x - image_size // 2, y - image_size // 2,
                                  frame.scaled(image_size, image_size,
                                               Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        # 只在窗口尺寸变化时重绘背景
        self._update_bg_pixmap()
        super().resizeEvent(event)

    def _update_bg_pixmap(self):
        rect = self.rect()
        margin = 20
        map_width = rect.width() - 2 * margin
        map_height = rect.height() - 2 * margin
        if map_width <= 0 or map_height <= 0:
            return
        # 创建 QPixmap 并绘制背景贴图和网格
        self.bg_pixmap = QPixmap(self.size())
        self.bg_pixmap.fill(QColor(255, 255, 255, 0))  # 透明底
        painter = QPainter(self.bg_pixmap)
        # 绘制背景贴图
        painter.drawImage(margin, margin, self.bg_image.scaled(map_width, map_height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
        # 绘制网格
        grid_size = 20
        pen = QPen(QColor(200, 200, 200, 50), 1)
        painter.setPen(pen)
        for x in range(0, map_width, grid_size):
            painter.drawLine(margin + x, margin, margin + x, margin + map_height)
        for y in range(0, map_height, grid_size):
            painter.drawLine(margin, margin + y, margin + map_width, margin + y)
        painter.end()

    def paintEvent(self, event):
        if not self.render:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # 先画缓存的背景
        if self.bg_pixmap is None or self.bg_pixmap.size() != self.size():
            self._update_bg_pixmap()
        if self.bg_pixmap:
            painter.drawPixmap(0, 0, self.bg_pixmap)

        # 绘制地图边界
        rect = self.rect()
        margin = 20
        map_width = rect.width() - 2 * margin
        map_height = rect.height() - 2 * margin

        # 绘制背景
        # painter.fillRect(margin, margin, map_width, map_height, QColor(240, 240, 240))

        # # 绘制网格
        # grid_size = 2000
        # pen = QPen(QColor(200, 200, 200), 1)
        # painter.setPen(pen)
        #
        # for x in range(0, map_width, grid_size):
        #     painter.drawLine(margin + x, margin, margin + x, margin + map_height)
        #
        # for y in range(0, map_height, grid_size):
        #     painter.drawLine(margin, margin + y, margin + map_width, margin + y)

        # 绘制障碍物（黑色方块）
        if hasattr(self.env, 'obstacles'):
            for obstacle in self.env.obstacles:
                # 缩放到画布尺寸
                x = margin + int(obstacle['x'] * map_width / self.env.map_size[0])
                y = margin + int(obstacle['y'] * map_height / self.env.map_size[1])
                radius = int(obstacle['radius'] * map_width / self.env.map_size[0])

                painter.setPen(QPen(Qt.black, 2))
                painter.setBrush(QColor(0, 0, 0))
                painter.drawRect(x - radius, y - radius, radius * 2, radius * 2)

        # 检查是否有新的爆炸需要添加
        self.check_for_new_explosions()

        # 绘制爆炸动画
        self.draw_explosions(painter, margin, map_width, map_height)
        # 绘制目标点（使用图片）
        if hasattr(self.env, 'targets'):
            for target in self.env.targets:
                if target['active']:
                    # 缩放到画布尺寸
                    x = margin + int(target['x'] * map_width / self.env.map_size[0])
                    y = margin + int(target['y'] * map_height / self.env.map_size[1])

                    attack_range = int(50 * map_width / self.env.map_size[0])
                    painter.setPen(QPen(QColor(200, 0, 0, 255), 1))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawEllipse(x - attack_range, y - attack_range,
                                        attack_range * 2, attack_range * 2)

                    # 计算图片尺寸
                    # image_size = int(28000 / self.env.map_size[0])
                    # # 确保图片尺寸不会过小或过大
                    # min_size = 30
                    # max_size = 50
                    # image_size = max(min_size, min(image_size, max_size))  # 图片显示大小
                    image_size = 20
                    # 保存当前画笔状态
                    painter.save()
                    # 移动坐标系到目标点中心
                    painter.translate(x, y)
                    painter.rotate(math.degrees(math.pi / 2 - target['theta']))
                    # 绘制目标点图片（居中显示）
                    painter.drawImage(-image_size / 2, -image_size / 2,
                                      self.target_image.scaled(image_size, image_size,
                                                          Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    # 恢复画笔状态
                    painter.restore()
                    # 绘制飞机状态信息
                    status_text = f"HP={target['blood']}"
                    painter.setPen(QPen(Qt.black, 1))
                    painter.drawText(x - 30, y - 20, status_text)

        # 绘制飞机
        if hasattr(self.env, 'agents'):
            for i, agent in enumerate(self.env.agents):
                if agent['dead']:
                    continue

                # 缩放到画布尺寸
                x = margin + int(agent['x'] * map_width / self.env.map_size[0])
                y = margin + int(agent['y'] * map_height / self.env.map_size[1])

                # 绘制攻击范围（蓝色圆圈）
                attack_range = int(agent['attack_range'] * map_width / self.env.map_size[0])
                painter.setPen(QPen(QColor(0, 0, 200, 255), 1))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(x - attack_range, y - attack_range,
                                    attack_range * 2, attack_range * 2)

                # 绘制飞机（使用图片）
                # 计算图片尺寸
                # image_size = int(28000 / self.env.map_size[0])
                # # 确保图片尺寸不会过小或过大
                # min_size = 30
                # max_size = 50
                image_size = 20
                # image_size = max(min_size, min(image_size, max_size))  # 图片显示大小
                # 保存当前画笔状态
                painter.save()
                # 移动坐标系到飞机中心
                painter.translate(x, y)
                # 旋转坐标系到飞机朝向
                painter.rotate(math.degrees(math.pi / 2 - agent['theta']))
                # 绘制飞机图片（居中显示）
                painter.drawImage(-image_size / 2, -image_size / 2,
                                  self.plane_image.scaled(image_size, image_size,
                                                     Qt.KeepAspectRatio, Qt.SmoothTransformation))
                # 恢复画笔状态
                painter.restore()
                # 绘制飞机编号
                painter.setPen(QPen(Qt.white, 1))
                painter.setFont(QFont('Arial', 8))
                # painter.drawText(x - 5, y + 5, str(i + 1))
                # 绘制飞机状态信息
                status_text = f"#{i + 1}: HP={agent['blood']}, AMMO={agent['bulleted_num']}"
                painter.setPen(QPen(Qt.black, 1))
                painter.drawText(x - 30, y - 20, status_text)

    def update_canvas(self):
        if self.render:
            self.update()

    def repaint_canvas(self):
        # for agent, pos in zip(self., positions):
        #     anim = QPropertyAnimation(agent, b"pos")
        #     anim.setDuration(1000)
        #     anim.setStartValue(agent.pos())
        #     anim.setEndValue(QPointF(*pos))
        #     anim.start()
        if self.render:
            self.repaint()


class RewardsCanvas(FigureCanvas):
    """用于展示多智能体训练过程的可视化画布"""
    def __init__(self, num_agents, width=12, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        # 配置网格布局
        self.gs = gridspec.GridSpec(2, 2, figure=self.fig,
                                    height_ratios=[1, 1], width_ratios=[1, 1],
                                    wspace=0.2, hspace=0.3)

        # 创建子图
        self.rewards_ax = self.fig.add_subplot(self.gs[0, :])  # 占满第一行
        self.actor_loss_ax = self.fig.add_subplot(self.gs[1, 0])  # 第二行左图
        self.critic_loss_ax = self.fig.add_subplot(self.gs[1, 1])  # 第二行右图

        # 数据存储
        self.num_agents = num_agents
        self.rewards_history = []
        self.actor_loss_history = [[] for _ in range(num_agents)]
        self.critic_loss_history = [[] for _ in range(num_agents)]
        self.entropy_history = [[] for _ in range(num_agents)]  # 新增熵历史记录

        # 初始化图表
        self._init_plots()

        # 智能体颜色映射
        self.agent_colors = list(mcolors.TABLEAU_COLORS.values())
        if num_agents > len(self.agent_colors):
            # 如果智能体数量超过预设颜色，生成更多颜色
            cmap = plt.cm.get_cmap('tab20', num_agents)
            self.agent_colors = [cmap(i) for i in range(num_agents)]

        self.fig.tight_layout()

    def _init_plots(self):
        """初始化所有图表"""
        # 奖励图表
        self.rewards_ax.set_title('智能体累积奖励')
        self.rewards_ax.set_xlabel('回合')
        self.rewards_ax.set_ylabel('累积奖励')
        self.rewards_ax.grid(True, linestyle='--', alpha=0.7)

        # Actor损失图表
        self.actor_loss_ax.set_title('Actor网络损失')
        self.actor_loss_ax.set_xlabel('训练步数')
        self.actor_loss_ax.set_ylabel('损失值')
        self.actor_loss_ax.grid(True, linestyle='--', alpha=0.7)
        self.actor_loss_ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Critic损失图表
        self.critic_loss_ax.set_title('Critic网络损失')
        self.critic_loss_ax.set_xlabel('训练步数')
        self.critic_loss_ax.set_ylabel('损失值')
        self.critic_loss_ax.grid(True, linestyle='--', alpha=0.7)
        self.critic_loss_ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    def update_plot(self, rewards, actor_losses, critic_losses, entropies=None):
        """更新所有图表数据"""
        # 更新奖励历史
        self.rewards_history.append(rewards)

        # 更新损失历史
        for i in range(self.num_agents):
            self.actor_loss_history[i].append(actor_losses[i])
            self.critic_loss_history[i].append(critic_losses[i])
            if entropies is not None:
                self.entropy_history[i].append(entropies[i])

        # 刷新图表
        self._update_rewards_plot()
        self._update_actor_loss_plot()
        self._update_critic_loss_plot()

        with open('actor_loss_history.pkl', 'wb') as file:
            pickle.dump(self.actor_loss_history, file)
        with open('rewards_history.pkl', 'wb') as file:
            pickle.dump(self.rewards_history, file)
        with open('critic_loss_history.pkl', 'wb') as file:
            pickle.dump(self.critic_loss_history, file)

        self.fig.tight_layout()
        self.draw()

    def _update_rewards_plot(self):
        """更新奖励图表"""
        self.rewards_ax.clear()
        self.rewards_ax.set_title('智能体累积奖励')
        self.rewards_ax.set_xlabel('回合')
        self.rewards_ax.set_ylabel('累积奖励')
        self.rewards_ax.grid(True, linestyle='--', alpha=0.7)

        # 绘制每个智能体的奖励
        for i in range(self.num_agents):
            if len(self.rewards_history) > 0:
                agent_rewards = [rh[i] for rh in self.rewards_history]
                line, = self.rewards_ax.plot(agent_rewards, label=f'智能体 {i + 1}',
                                             color=self.agent_colors[i], linewidth=1.5)

                # 添加最后一个点的标记和数值
                if len(agent_rewards) > 0:
                    self.rewards_ax.scatter(len(agent_rewards) - 1, agent_rewards[-1],
                                            color=line.get_color(), s=50)
                    self.rewards_ax.annotate(f'{agent_rewards[-1]:.1f}',
                                             xy=(len(agent_rewards) - 1, agent_rewards[-1]),
                                             xytext=(5, 5),
                                             textcoords='offset points',
                                             fontsize=9)

        # 计算并绘制平均奖励
        if len(self.rewards_history) > 0:
            mean_rewards = np.mean(self.rewards_history, axis=1)
            self.rewards_ax.plot(mean_rewards, label='平均奖励',
                                 color='black', linestyle='--', linewidth=2)

        self.rewards_ax.legend(loc='best')
        self.rewards_ax.set_xlim(left=0)
        self.rewards_ax.set_ylim(bottom=min([min(r) for r in self.rewards_history]) - 10
        if self.rewards_history else 0)

    def _update_actor_loss_plot(self):
        """更新Actor损失图表"""
        self.actor_loss_ax.clear()
        self.actor_loss_ax.set_title('Actor网络损失')
        self.actor_loss_ax.set_xlabel('训练步数')
        self.actor_loss_ax.set_ylabel('损失值')
        self.actor_loss_ax.grid(True, linestyle='--', alpha=0.7)

        # 绘制每个智能体的Actor损失
        for i in range(self.num_agents):
            if len(self.actor_loss_history[i]) > 0:
                self.actor_loss_ax.plot(self.actor_loss_history[i],
                                        label=f'智能体 {i + 1}',
                                        color=self.agent_colors[i],
                                        linewidth=1.5)

        self.actor_loss_ax.legend(loc='best')
        self.actor_loss_ax.set_xlim(left=0)

        # 自适应y轴范围
        all_losses = [loss for agent_losses in self.actor_loss_history for loss in agent_losses]
        if all_losses:
            self.actor_loss_ax.set_ylim(bottom=min(all_losses)-1,
                                        top=max(all_losses) * 1.05)

    def _update_critic_loss_plot(self):
        """更新Critic损失图表"""
        self.critic_loss_ax.clear()
        self.critic_loss_ax.set_title('Critic网络损失')
        self.critic_loss_ax.set_xlabel('训练步数')
        self.critic_loss_ax.set_ylabel('损失值')
        self.critic_loss_ax.grid(True, linestyle='--', alpha=0.7)

        # 绘制每个智能体的Critic损失
        for i in range(self.num_agents):
            if len(self.critic_loss_history[i]) > 0:
                self.critic_loss_ax.plot(self.critic_loss_history[i],
                                         label=f'智能体 {i + 1}',
                                         color=self.agent_colors[i],
                                         linewidth=1.5)

        self.critic_loss_ax.legend(loc='best')
        self.critic_loss_ax.set_xlim(left=0)

        # 自适应y轴范围
        all_losses = [loss for agent_losses in self.critic_loss_history for loss in agent_losses]
        if all_losses:
            self.critic_loss_ax.set_ylim(bottom=min(all_losses)-1,
                                         top=max(all_losses) * 1.05)


class AgentStatusWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.status_labels = []
        for i in range(5):  # 最多显示5个智能体
            label = QLabel(f"智能体 {i + 1}: 未初始化")
            self.status_labels.append(label)
            layout.addWidget(label)

        self.setLayout(layout)

    def update_status(self, env):
        if not hasattr(env, 'agent_positions'):
            return

        for i, pos in enumerate(env.agent_positions):
            if i < len(self.status_labels):
                self.status_labels[i].setText(f"智能体 {i + 1}: 位置=({pos[0]}, {pos[1]})")
