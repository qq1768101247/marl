import math
import sys

from PyQt5 import QtCore
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QComboBox, QLabel,
                             QTabWidget, QGroupBox, QGridLayout, QMessageBox, QSpinBox)
from PyQt5.QtCore import Qt, QThreadPool
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QImage
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from aircraft_env import AircraftEnv

from algorithm import QLearning, PPO
from MASAC import MASACAlgorithm
from QLearning_net.ContinuousQLearning import ContinuousQLearning
from mappo.algorithms.mappo import MAPPO

from training_thread import TrainingThread

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

                    try:
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

                    except Exception as e:
                        # 如果加载图片失败，回退到绘制绿色圆形
                        print(f"目标点图片加载失败: {e}")

                        painter.setPen(QPen(Qt.green, 2))
                        painter.setBrush(QColor(0, 255, 0, 100))  # 半透明
                        painter.drawEllipse(x - 10, y - 10, 20, 20)

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

                # 绘制飞机朝向
                heading_x = x + math.cos(agent['theta']) * 20
                heading_y = y + math.sin(agent['theta']) * 20

                # 绘制飞机（使用图片）
                try:
                    # 加载飞机图片
                    plane_image = QImage("resource/image/enemy1.png")
                    if plane_image.isNull():
                        raise Exception("无法加载飞机图片")

                    # 计算图片尺寸
                    image_size = int(5000 / self.env.map_size[0])

                    # 确保图片尺寸不会过小或过大
                    min_size = 20
                    max_size = 50
                    image_size = max(min_size, min(image_size, max_size)) # 图片显示大小

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

                except Exception as e:
                    # 如果加载图片失败，回退到绘制三角形
                    print(f"图片加载失败: {e}")

                    # 绘制飞机（蓝色三角形）
                    painter.setPen(QPen(QColor(0, 0, 255), 2))
                    painter.setBrush(QColor(0, 0, 255))

                    # 计算三角形三个顶点
                    angle1 = agent['theta']
                    angle2 = agent['theta'] + 2 * math.pi / 3
                    angle3 = agent['theta'] + 4 * math.pi / 3

                    size = 15
                    x1 = x + math.cos(angle1) * size
                    y1 = y + math.sin(angle1) * size
                    x2 = x + math.cos(angle2) * size
                    y2 = y + math.sin(angle2) * size
                    x3 = x + math.cos(angle3) * size
                    y3 = y + math.sin(angle3) * size

                    painter.drawPolygon([
                        QtCore.QPointF(x1, y1),
                        QtCore.QPointF(x2, y2),
                        QtCore.QPointF(x3, y3)
                    ])

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
            self.axes.plot(agent_rewards, label=f'智能体 {i+1}')
        
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
            label = QLabel(f"智能体 {i+1}: 未初始化")
            self.status_labels.append(label)
            layout.addWidget(label)
        
        self.setLayout(layout)

    def update_status(self, env):
        if not hasattr(env, 'agent_positions'):
            return

        for i, pos in enumerate(env.agent_positions):
            if i < len(self.status_labels):
                self.status_labels[i].setText(f"智能体 {i+1}: 位置=({pos[0]}, {pos[1]})")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('多智能体强化学习训练平台')
        self.setGeometry(100, 100, 1200, 800)
        self.num_agents = 5

        self.init_environment()
        self.init_algorithms()
        self.init_ui()

        self.training_thread = None
        self.is_training = False

        self.threadpool = QThreadPool()
        self.Render = eval(self.render_box.currentText())


    def init_environment(self):
        # 初始化环境
        self.environments = {
            "简单地图": AircraftEnv(map_size=(50, 50), num_agents=self.num_agents),
            "中等地图": AircraftEnv(map_size=(100, 100), num_agents=self.num_agents),
            "复杂地图": AircraftEnv(map_size=(150, 150), num_agents=self.num_agents)
        }
        self.current_env = self.environments["简单地图"]

    def init_algorithms(self):
        # 初始化算法
        self.algorithms = {
            "Q-learning": lambda env: QLearning(env),
            "Q-learning-net": lambda env: ContinuousQLearning(env),
            "PPO": lambda env: PPO(env),
            "MASAC": lambda env: MASACAlgorithm(env),
            "MAPPO": lambda env: MAPPO(env),
        }
        self.current_algorithm = self.algorithms["Q-learning"](self.current_env)

    def init_ui(self):
        # 创建主部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)

        # 左侧控制面板
        control_panel = QWidget()
        control_panel.setMaximumWidth(300)
        control_layout = QVBoxLayout(control_panel)

        # 算法选择
        algorithm_group = QGroupBox("选择算法")
        algorithm_layout = QVBoxLayout()

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(list(self.algorithms.keys()))
        self.algorithm_combo.currentTextChanged.connect(self.on_algorithm_changed)

        algorithm_layout.addWidget(QLabel("算法:"))
        algorithm_layout.addWidget(self.algorithm_combo)
        algorithm_group.setLayout(algorithm_layout)

        # 地图选择
        map_group = QGroupBox("选择地图")
        map_layout = QVBoxLayout()

        self.map_combo = QComboBox()
        self.map_combo.addItems(list(self.environments.keys()))
        self.map_combo.currentTextChanged.connect(self.on_map_changed)

        map_layout.addWidget(QLabel("地图:"))
        map_layout.addWidget(self.map_combo)
        map_group.setLayout(map_layout)

        # 智能体数量选择
        agents_group = QGroupBox("智能体设置")
        agents_layout = QVBoxLayout()

        self.agents_label = QLabel("智能体数量:")
        self.agents_input = QSpinBox()
        self.agents_input.setRange(1, 50)
        self.agents_input.setValue(self.num_agents)
        self.agents_input.valueChanged.connect(self.on_agents_changed)

        agents_layout.addWidget(self.agents_label)
        agents_layout.addWidget(self.agents_input)
        agents_group.setLayout(agents_layout)

        # 训练参数
        params_group = QGroupBox("训练参数")
        params_layout = QGridLayout()

        self.render_label = QLabel("是否渲染")
        self.render_box = QComboBox()
        self.render_box.addItems(['True', 'False'])
        self.render_box.setCurrentText("100")
        self.render_box.currentTextChanged.connect(self.on_render_changed)

        self.episodes_label = QLabel("训练回合数:")
        self.episodes_input = QComboBox()
        self.episodes_input.addItems(["50", "100", "200", "500", "1000"])
        self.episodes_input.setCurrentText("100")

        params_layout.addWidget(self.render_label, 0, 0)
        params_layout.addWidget(self.render_box, 0, 1)
        params_layout.addWidget(self.episodes_label, 1, 0)
        params_layout.addWidget(self.episodes_input, 1, 1)
        params_group.setLayout(params_layout)

        # 算法超参数显示区域
        self.hyperparams_group = QGroupBox("算法超参数")
        self.hyperparams_layout = QVBoxLayout()
        self.hyperparams_group.setLayout(self.hyperparams_layout)

        # 控制按钮
        button_layout = QHBoxLayout()

        self.start_button = QPushButton("开始训练")
        self.start_button.clicked.connect(self.on_start_training)

        self.stop_button = QPushButton("停止训练")
        self.stop_button.clicked.connect(self.on_stop_training)
        self.stop_button.setEnabled(False)

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)

        # 添加到控制面板
        control_layout.addWidget(algorithm_group)
        control_layout.addWidget(map_group)
        control_layout.addWidget(agents_group)
        control_layout.addWidget(params_group)
        control_layout.addWidget(self.hyperparams_group)
        control_layout.addLayout(button_layout)
        control_layout.addStretch()

        # 右侧显示区域
        display_panel = QWidget()
        display_layout = QVBoxLayout(display_panel)

        # 创建标签页
        self.tabs = QTabWidget()

        # 地图显示标签页
        self.map_tab = QWidget()
        self.map_tab_layout = QVBoxLayout(self.map_tab)
        self.map_canvas = MapCanvas(self.current_env)
        self.map_tab_layout.addWidget(self.map_canvas)
        self.tabs.addTab(self.map_tab, "地图显示")

        # 智能体状态标签页
        self.status_tab = QWidget()
        self.status_tab_layout = QVBoxLayout(self.status_tab)
        self.agent_status = AgentStatusWidget()
        self.status_tab_layout.addWidget(self.agent_status)
        self.tabs.addTab(self.status_tab, "智能体状态")

        # 奖励曲线标签页
        self.rewards_tab = QWidget()
        self.rewards_tab_layout = QVBoxLayout(self.rewards_tab)
        self.rewards_canvas = RewardsCanvas()
        self.rewards_tab_layout.addWidget(self.rewards_canvas)
        self.tabs.addTab(self.rewards_tab, "奖励曲线")

        display_layout.addWidget(self.tabs)

        # 添加到主布局
        main_layout.addWidget(control_panel)
        main_layout.addWidget(display_panel, 1)

        # 初始化超参数显示
        self.show_hyperparams()

    def on_agents_changed(self, value):
        self.num_agents = value
        self.init_environment()
        self.current_env = self.environments[self.map_combo.currentText()]
        self.map_canvas.env = self.current_env
        self.map_canvas.update_canvas()
        self.agent_status.update_status(self.current_env)
        self.current_algorithm = self.algorithms[self.algorithm_combo.currentText()](self.current_env)
        print(f"智能体数量已设置为: {self.num_agents}")

    def on_algorithm_changed(self, algorithm_name):
        self.current_algorithm = self.algorithms[algorithm_name](self.current_env)
        print(f"已选择算法: {algorithm_name}")
        self.show_hyperparams()

    def on_map_changed(self, map_name):
        self.current_env = self.environments[map_name]
        self.map_canvas.env = self.current_env
        self.map_canvas.update_canvas()
        self.agent_status.update_status(self.current_env)
        self.current_algorithm = self.algorithms[self.algorithm_combo.currentText()](self.current_env)
        print(f"已选择地图: {map_name}")
        self.show_hyperparams()

    def on_render_changed(self, render_name):
        self.Render = eval(render_name)

    def on_start_training(self):
        if self.is_training:
            return

        if self.current_algorithm is None:
            QMessageBox.warning(self, "警告", "请先选择算法")
            return

        self.is_training = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        # 获取训练回合数
        num_episodes = int(self.episodes_input.currentText())

        # 创建训练线程
        self.training_thread = TrainingThread(self.current_env, self.current_algorithm, num_episodes)
        self.training_thread.update_signal.connect(self.on_training_update)
        self.training_thread.finished_signal.connect(self.on_training_finished)
        self.training_thread.start()

    def on_stop_training(self):
        if self.training_thread and self.is_training:
            self.training_thread.running = False
            self.is_training = False
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def on_training_update(self, data):
        # 更新智能体状态
        if self.Render:
            self.agent_status.update_status(self.current_env)
            # 更新地图显示
            self.map_canvas.repaint_canvas(data["positions"])

        # 更新奖励曲线
        if 'rewards' in data:
            self.rewards_canvas.update_plot(data['rewards'])

    def on_training_finished(self, data):
        if self.training_thread:
            self.training_thread = False
        self.is_training = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        QMessageBox.information(self, "训练完成", "训练已完成!")

    def show_hyperparams(self):
        # 清空之前的超参数显示
        for i in reversed(range(self.hyperparams_layout.count())):
            self.hyperparams_layout.itemAt(i).widget().setParent(None)

        algorithm_name = self.algorithm_combo.currentText()
        if algorithm_name == "Q-learning":
            alpha = self.current_algorithm.alpha
            gamma = self.current_algorithm.gamma
            epsilon = self.current_algorithm.epsilon
            num_actions_per_dim = self.current_algorithm.num_actions_per_dim
            self.hyperparams_layout.addWidget(QLabel(f"学习率 (alpha): {alpha}"))
            self.hyperparams_layout.addWidget(QLabel(f"折扣因子 (gamma): {gamma}"))
            self.hyperparams_layout.addWidget(QLabel(f"探索率 (epsilon): {epsilon}"))
            self.hyperparams_layout.addWidget(QLabel(f"每个维度的离散动作数量: {num_actions_per_dim}"))
        elif algorithm_name == "MASAC":
            gamma = self.current_algorithm.gamma
            batch_size = self.current_algorithm.batch_size
            memory_capacity = self.current_algorithm.memory.capacity
            self.hyperparams_layout.addWidget(QLabel(f"折扣因子 (gamma): {gamma}"))
            self.hyperparams_layout.addWidget(QLabel(f"批量大小 (batch_size): {batch_size}"))
            self.hyperparams_layout.addWidget(QLabel(f"经验回放缓冲区容量: {memory_capacity}"))
        elif algorithm_name == "Q-learning-net":
            lr = self.current_algorithm.lr
            gamma = self.current_algorithm.gamma
            epsilon = self.current_algorithm.epsilon
            self.hyperparams_layout.addWidget(QLabel(f"学习率 (lr): {lr}"))
            self.hyperparams_layout.addWidget(QLabel(f"折扣因子 (gamma): {gamma}"))
            self.hyperparams_layout.addWidget(QLabel(f"探索率 (epsilon): {epsilon}"))
        elif algorithm_name == "MAPPO":
            lr = self.current_algorithm.lr
            gamma = self.current_algorithm.gamma
            clip_epsilon = self.current_algorithm.clip_epsilon
            ent_coef = self.current_algorithm.ent_coef
            vf_coef = self.current_algorithm.vf_coef
            gae_lambda = self.current_algorithm.gae_lambda
            max_grad_norm = self.current_algorithm.max_grad_norm
            batch_size = self.current_algorithm.batch_size
            buffer_size = self.current_algorithm.buffer_size
            self.hyperparams_layout.addWidget(QLabel(f"(lr): {lr}"))
            self.hyperparams_layout.addWidget(QLabel(f"(gamma): {gamma}"))
            self.hyperparams_layout.addWidget(QLabel(f"(clip_epsilon): {clip_epsilon}"))
            self.hyperparams_layout.addWidget(QLabel(f"(ent_coef): {ent_coef}"))
            self.hyperparams_layout.addWidget(QLabel(f"(vf_coef): {vf_coef}"))
            self.hyperparams_layout.addWidget(QLabel(f"(gae_lambda): {gae_lambda}"))
            self.hyperparams_layout.addWidget(QLabel(f"(max_grad_norm): {max_grad_norm}"))
            self.hyperparams_layout.addWidget(QLabel(f"(batch_size): {batch_size}"))
            self.hyperparams_layout.addWidget(QLabel(f"(buffer_size): {buffer_size}"))
        # 可以继续添加其他算法的超参数显示

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())    