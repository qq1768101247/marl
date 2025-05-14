import math

import matplotlib
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QImage
from PyQt5.QtWidgets import (QWidget, QVBoxLayout,
                             QLabel)

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False


class MapCanvas(QWidget):
    def __init__(self, env, parent=None):
        super().__init__(parent)
        self.env = env
        self.setMinimumSize(400, 400)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制地图边界
        rect = self.rect()
        margin = 20
        map_width = rect.width() - 2 * margin
        map_height = rect.height() - 2 * margin

        # 绘制背景
        painter.fillRect(margin, margin, map_width, map_height, QColor(240, 240, 240))

        # 绘制网格
        grid_size = 20
        pen = QPen(QColor(200, 200, 200), 1)
        painter.setPen(pen)

        for x in range(0, map_width, grid_size):
            painter.drawLine(margin + x, margin, margin + x, margin + map_height)

        for y in range(0, map_height, grid_size):
            painter.drawLine(margin, margin + y, margin + map_width, margin + y)

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

        # 绘制目标点（使用图片）
        if hasattr(self.env, 'targets'):
            for target in self.env.targets:
                if target['active']:
                    # 缩放到画布尺寸
                    x = margin + int(target['x'] * map_width / self.env.map_size[0])
                    y = margin + int(target['y'] * map_height / self.env.map_size[1])
                    # 加载目标点图片
                    target_image = QImage("resource/image/enemy2.png")
                    if target_image.isNull():
                        raise Exception("无法加载目标点图片")
                    # 计算图片尺寸
                    image_size = int(5000 / self.env.map_size[0])
                    # 确保图片尺寸不会过小或过大
                    min_size = 20
                    max_size = 50
                    image_size = max(min_size, min(image_size, max_size))  # 图片显示大小
                    # 保存当前画笔状态
                    painter.save()
                    # 移动坐标系到目标点中心
                    painter.translate(x, y)
                    # 绘制目标点图片（居中显示）
                    painter.drawImage(-image_size / 2, -image_size / 2,
                                      target_image.scaled(image_size, image_size,
                                                          Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    # 恢复画笔状态
                    painter.restore()

        # 绘制飞机
        if hasattr(self.env, 'agents'):
            for i, agent in enumerate(self.env.agents):
                if agent['dead']:
                    continue

                # 缩放到画布尺寸
                x = margin + int(agent['x'] * map_width / self.env.map_size[0])
                y = margin + int(agent['y'] * map_height / self.env.map_size[1])

                # 绘制攻击范围（红色圆圈）
                attack_range = int(agent['attack_range'] * map_width / self.env.map_size[0])
                painter.setPen(QPen(QColor(255, 0, 0, 100), 1))  # 半透明红色
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(x - attack_range, y - attack_range,
                                    attack_range * 2, attack_range * 2)

                # 绘制飞机（使用图片）
                # 加载飞机图片
                plane_image = QImage("resource/image/enemy1.png")
                if plane_image.isNull():
                    raise Exception("无法加载飞机图片")
                # 计算图片尺寸
                image_size = int(5000 / self.env.map_size[0])
                # 确保图片尺寸不会过小或过大
                min_size = 20
                max_size = 50
                image_size = max(min_size, min(image_size, max_size))  # 图片显示大小
                # 保存当前画笔状态
                painter.save()
                # 移动坐标系到飞机中心
                painter.translate(x, y)
                # 旋转坐标系到飞机朝向
                painter.rotate(math.degrees(agent['theta'] + math.pi / 2))
                # 绘制飞机图片（居中显示）
                painter.drawImage(-image_size / 2, -image_size / 2,
                                  plane_image.scaled(image_size, image_size,
                                                     Qt.KeepAspectRatio, Qt.SmoothTransformation))
                # 恢复画笔状态
                painter.restore()
                # 绘制飞机编号
                painter.setPen(QPen(Qt.white, 1))
                painter.setFont(QFont('Arial', 8))
                painter.drawText(x - 5, y + 5, str(i + 1))
                # 绘制飞机状态信息
                status_text = f"#{i + 1}: HP={agent['blood']}, AMMO={agent['volume']}"
                painter.setPen(QPen(Qt.black, 1))
                painter.drawText(x - 30, y - 20, status_text)

    def update_canvas(self):
        self.update()

    def repaint_canvas(self, positions):
        # for agent, pos in zip(self., positions):
        #     anim = QPropertyAnimation(agent, b"pos")
        #     anim.setDuration(1000)
        #     anim.setStartValue(agent.pos())
        #     anim.setEndValue(QPointF(*pos))
        #     anim.start()
        self.repaint()


class RewardsCanvas(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_title('智能体奖励')
        self.axes.set_xlabel('回合')
        self.axes.set_ylabel('平均奖励')
        self.fig.tight_layout()
        self.rewards_history = []

    def update_plot(self, rewards):
        self.rewards_history.append(rewards)
        self.axes.clear()
        self.axes.set_title('智能体奖励')
        self.axes.set_xlabel('回合')
        self.axes.set_ylabel('平均奖励')

        # 绘制每个智能体的奖励
        num_agents = len(rewards)
        for i in range(num_agents):
            agent_rewards = [rh[i] for rh in self.rewards_history]
            self.axes.plot(agent_rewards, label=f'智能体 {i + 1}')

        self.axes.legend()
        self.fig.tight_layout()
        self.draw()


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
