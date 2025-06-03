import sys

import matplotlib
from PyQt5.QtCore import QThreadPool
from PyQt5.QtWidgets import (QApplication, QMainWindow, QHBoxLayout, QPushButton, QComboBox, QTabWidget, QGroupBox,
                             QGridLayout, QMessageBox, QSpinBox, QFrame)

from evaluation_thread import EvaluationThread

matplotlib.use('Qt5Agg')
from aircraft_env import AircraftEnv

from algorithm import QLearning, PPO
from MASAC import MASACAlgorithm
from QLearning_net.ContinuousQLearning import ContinuousQLearning
from mappo.algorithms.mappo import MAPPO

from training_thread import TrainingThread

from mapcanvas import *
# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

class AgentStatusWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)

        self.status_frame = QFrame()
        self.status_frame.setStyleSheet("""
            QFrame {
                background-color: #112240;
                border-radius: 8px;
                padding: 10px;
            }
            QLabel {
                color: #CCD6F6;
                font-size: 14px;
                padding: 5px;
            }
        """)

        status_layout = QVBoxLayout(self.status_frame)
        self.status_labels = []

        for i in range(6):  # 最多显示5个智能体
            label = QLabel(f"智能体 {i + 1}: 未初始化")
            label.setStyleSheet("border-bottom: 1px solid #303C55;")
            self.status_labels.append(label)
            status_layout.addWidget(label)

        layout.addWidget(QLabel("智能体状态:"))
        layout.addWidget(self.status_frame)
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
        self.Render = False
        self.setWindowTitle('多智能体强化学习训练平台')
        self.setGeometry(100, 100, 1200, 800)
        self.num_agents = 6
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0A192F;
            }
            QGroupBox {
                background-color: #112240;
                border: 1px solid #303C55;
                border-radius: 8px;
                margin-top: 10px;
                color: #CCD6F6;
                font-size: 16px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 10px;
                background-color: #0A192F;
                border-radius: 5px;
            }
            QLabel {
                color: #CCD6F6;
                font-size: 14px;
            }
            QComboBox {
                background-color: #1E3A8A;
                color: #CCD6F6;
                border: 1px solid #3B82F6;
                border-radius: 4px;
                padding: 5px;
                min-height: 25px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left-width: 1px;
                border-left-color: #3B82F6;
                border-left-style: solid;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }
            QSpinBox {
                background-color: #1E3A8A;
                color: #CCD6F6;
                border: 1px solid #3B82F6;
                border-radius: 4px;
                padding: 5px;
                min-height: 25px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 16px;
                border-left: 1px solid #3B82F6;
            }
            QPushButton {
                background-color: #3B82F6;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #2563EB;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
            QPushButton:disabled {
                background-color: #60A5FA;
                color: #DBEAFE;
            }
            QTabWidget::pane {
                border: 1px solid #303C55;
                border-radius: 8px;
                background-color: #0A192F;
                margin-top: -2px;
            }
            QTabBar::tab {
                background-color: #112240;
                color: #CCD6F6;
                border: 1px solid #303C55;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #0A192F;
                color: #3B82F6;
                font-weight: bold;
                border-color: #3B82F6;
                border-bottom: 1px solid #0A192F;
                margin-bottom: -1px;
            }
            QTabBar::tab:hover:not(:selected) {
                background-color: #1E3A8A;
            }
        """)


        self.init_environment()
        self.init_algorithms()
        self.init_ui()

        self.training_thread = None
        self.is_training = False
        self.evaluation_thread = None

        self.threadpool = QThreadPool()



    def init_environment(self):
        # 初始化环境
        self.environments = {
            "简单地图": AircraftEnv(num_agents=self.num_agents),
            "中等地图": AircraftEnv(num_agents=self.num_agents),
            "复杂地图": AircraftEnv(num_agents=self.num_agents)
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
        self.agents_input.setRange(1, 16)
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
        self.Render = eval(self.render_box.currentText())

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

        self.eva_button = QPushButton("Evaluation")
        self.eva_button.clicked.connect(self.on_start_evaluation)

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.eva_button)

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
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #303C55;
                border-radius: 8px;
                background-color: #0A192F;
                padding: 10px;
            }
            QTabBar::tab {
                background-color: #112240;
                color: #CCD6F6;
                border: 1px solid #303C55;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #0A192F;
                color: #3B82F6;
                font-weight: bold;
                border-color: #3B82F6;
                border-bottom: 1px solid #0A192F;
                margin-bottom: -1px;
            }
            QTabBar::tab:hover:not(:selected) {
                background-color: #1E3A8A;
            }
        """)

        # 地图显示标签页
        self.map_tab = QWidget()
        self.map_tab_layout = QVBoxLayout(self.map_tab)
        self.map_canvas = MapCanvas(self.current_env, self.Render)
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
        self.rewards_canvas = RewardsCanvas(self.num_agents)
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
        self.map_canvas.set_render(self.Render)

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

    def on_start_evaluation(self):
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.eva_button.setEnabled(False)
        num_episodes = int(self.episodes_input.currentText())

        self.evaluation_thread = EvaluationThread(self.current_env, self.current_algorithm, num_episodes)
        self.evaluation_thread.update_signal.connect(self.on_training_update)
        self.evaluation_thread.finished_signal.connect(self.on_training_finished)
        self.evaluation_thread.start()
        pass
    def on_stop_training(self):
        if self.training_thread and self.is_training:
            self.training_thread.running = False
            self.is_training = False
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
        if self.evaluation_thread:
            self.evaluation_thread.running = False
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)


    def on_training_update(self, data):
        # 更新智能体状态
        if self.Render:
            self.agent_status.update_status(self.current_env)
            # 更新地图显示
            # self.map_canvas.repaint_canvas(data["positions"])
        # 更新奖励曲线
        if 'rewards' in data:
            self.rewards_canvas.update_plot(data['rewards'], data['actor_losses'],
                                            data['critic_losses'], data['entropies'])

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