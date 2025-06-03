from PyQt5.QtCore import QThread, pyqtSignal, QRunnable
import numpy as np
import time


class TrainingThread(QThread):
    update_signal = pyqtSignal(object)
    finished_signal = pyqtSignal(object)

    def __init__(self, env, algorithm, num_episodes=10000):
        super().__init__()
        self.env = env
        self.algorithm = algorithm
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

                # 更新算法
                if hasattr(self.algorithm, 'update'):
                    self.algorithm.update(state, actions, rewards, next_state, dones)

                state = next_state
                done = all(dones) or all(truncated)

                step += 1

                if step % self.env.max_steps == 0:
                    break
                if step % 100 == 0:
                    print(f"Episode {episode + 1}/{self.num_episodes}, {step=}")

            rewards_history.append(episode_reward)

            # 每个回合结束后保存模型
            self.algorithm.save_models()
            # 获取最新的损失值
            actor_losses = [0] * self.env.num_agents  # 占位，实际应从算法中获取
            critic_losses = [0] * self.env.num_agents  # 占位，实际应从算法中获取
            entropies = [0] * self.env.num_agents  # 占位，实际应从算法中获取

            # 如果算法提供了获取损失的方法，则使用它
            if hasattr(self.algorithm, 'get_losses'):
                actor_losses, critic_losses = self.algorithm.get_losses()

            # 如果算法提供了获取熵的方法，则使用它
            if hasattr(self.algorithm, 'get_entropies'):
                entropies = self.algorithm.get_entropies()

            self.update_signal.emit({
                'episode': episode + 1,
                'step': step,
                'positions': self.env.agent_positions,
                'rewards': episode_reward/step,
                'total_rewards': episode_reward,
                'actor_losses': actor_losses,
                'critic_losses': critic_losses,
                'entropies': entropies  # 新增熵数据
            })

        self.running = False
        print("Finished training")
        # 训练完成
        self.finished_signal.emit({
            'rewards_history': rewards_history
        })    