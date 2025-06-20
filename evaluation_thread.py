from PyQt5.QtCore import QThread, pyqtSignal, QRunnable
import numpy as np
import time

class EvaluationThread(QThread):
    update_signal = pyqtSignal(object)
    finished_signal = pyqtSignal(object)
    def __init__(self, current_env, current_algorithm, num_episodes):
        super().__init__()
        self.env = current_env
        self.algorithm = current_algorithm
        self.num_episodes = num_episodes
        self.running = True
        self.algorithm.load_models()

    def run(self):
        rewards_history = []
        for episode in range(self.num_episodes):
            if not self.running:
                break
            state, _ = self.env.reset()
            episode_reward = np.zeros(self.env.num_agents)
            done = False
            step = 0
            while not done:
                if not self.running:
                    break

                # 获取所有智能体的动作
                actions = [self.algorithm.get_action(i, state) for i in range(self.env.num_agents)]

                # 执行动作
                next_state, rewards, dones, truncated, _ = self.env.step(actions)
                episode_reward += rewards

                state = next_state
                done = all(dones) or all(truncated)

                # 更新UI
                step += 1

                if step % 100 == 0:
                    print(f"Episode {episode + 1}/{self.num_episodes}, {step=}")
                # 控制训练速度
                # time.sleep(0.001)

            rewards_history.append(episode_reward)

            # # 每个回合结束后更新UI
            # self.update_signal.emit({
            #     'episode': episode + 1,
            #     'step': step,
            #     'positions': self.env.agent_positions,
            #     'rewards': [0] * self.env.num_agents
            # })

            print(f"Episode {episode + 1}/{self.num_episodes}, "
                  f"Average Reward: {np.mean(episode_reward):.2f}")
        self.running = False
        print("finished training")
        # 训练完成
        self.finished_signal.emit({
            'rewards_history': rewards_history
        })
