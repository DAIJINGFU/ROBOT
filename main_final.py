import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os

# ==========================================
# 1. 环境定义 (对标：室内服务机器人/协同调度)
# ==========================================
class RobotGridEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size=10, num_robots=2, num_tasks=6, render_mode=None, fixed_tasks=True):
        self.grid_size = grid_size
        self.num_robots = num_robots
        self.num_tasks = num_tasks  
        self.render_mode = render_mode
        self.fixed_tasks = fixed_tasks
        
        self.observation_space = spaces.Dict({
            "robots": spaces.Box(0, grid_size, shape=(num_robots, 2), dtype=np.int32),
            "tasks": spaces.Box(0, grid_size, shape=(num_tasks, 2), dtype=np.int32),
            "task_status": spaces.MultiBinary(num_tasks) 
        })

        self.action_space = spaces.MultiDiscrete([5] * num_robots)
        
        # 任务分布：为了体现协同，任务均匀分布在两边
        # 左边3个，右边3个
        if fixed_tasks:
            self._fixed_task_locations = np.array([
                # 左半区 (给 Agent 0)
                [1, 1], [1, 8], [4, 2],
                # 右半区 (给 Agent 1)
                [8, 1], [8, 8], [5, 7]
            ][:num_tasks])
        else:
            self._fixed_task_locations = None

        self._action_to_direction = {
            0: np.array([0, 1]), 1: np.array([0, -1]),
            2: np.array([1, 0]), 3: np.array([-1, 0]),
            4: np.array([0, 0]), 
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 机器人出生在中间
        self._robot_locations = np.array([[4, 4], [5, 5]], dtype=np.int32)
        
        if self.fixed_tasks and self._fixed_task_locations is not None:
             self._task_locations = self._fixed_task_locations.copy()
        else:
             self._task_locations = self.np_random.integers(0, self.grid_size, size=(self.num_tasks, 2), dtype=np.int32)
             
        self._task_status = np.ones(self.num_tasks, dtype=np.int8)
        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "robots": self._robot_locations.copy(),
            "tasks": self._task_locations.copy(),
            "task_status": self._task_status.copy()
        }

    def step(self, action):
        rewards = np.zeros(self.num_robots)
        terminated = False
        truncated = False
        
        # 1. 移动
        for i in range(self.num_robots):
            direction = self._action_to_direction[action[i]]
            new_pos = self._robot_locations[i] + direction
            new_pos = np.clip(new_pos, 0, self.grid_size - 1)
            self._robot_locations[i] = new_pos
            
        # 2. 完成任务
        for t_idx in range(self.num_tasks):
            if self._task_status[t_idx] == 1:
                task_pos = self._task_locations[t_idx]
                for r_idx in range(self.num_robots):
                    if np.array_equal(self._robot_locations[r_idx], task_pos):
                        self._task_status[t_idx] = 0 
                        rewards[r_idx] += 20.0 # 完成任务给予高额奖励 (提高吞吐量)
                        break 
        
        # 3. 动态惩罚 (对标：等待时间优化)
        # 惩罚 = 基础消耗 + 系数 * 当前滞留任务数
        # 滞留任务越多，惩罚越重 -> 迫使机器人尽快清空队列
        active_tasks = np.sum(self._task_status)
        time_penalty = 0.1 + (0.05 * active_tasks) 
        rewards -= time_penalty 
        
        if active_tasks == 0:
            terminated = True
            
        return self._get_obs(), rewards, terminated, truncated, {}

# ==========================================
# 2. 算法定义 (对标：无模型多智能体RL)
# ==========================================
class QLearningAgent:
    def __init__(self, agent_id, action_space_size, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.agent_id = agent_id
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = {}

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(5)
        return np.argmax(self.q_table.get(state, np.zeros(5)))

    def learn(self, state, action, reward, next_state):
        current_q = self.q_table.get(state, np.zeros(5))
        next_q = self.q_table.get(next_state, np.zeros(5))
        # Q-Learning 更新公式
        current_q[action] += self.lr * (reward + self.gamma * np.max(next_q) - current_q[action])
        self.q_table[state] = current_q

    def save(self, is_best=False):
        suffix = "best" if is_best else "final"
        with open(f"agent_{self.agent_id}_{suffix}.pkl", 'wb') as f:
            pickle.dump(self.q_table, f)
            
    def load(self, is_best=True):
        suffix = "best" if is_best else "final"
        fname = f"agent_{self.agent_id}_{suffix}.pkl"
        if os.path.exists(fname):
            with open(fname, 'rb') as f:
                self.q_table = pickle.load(f)
            return True
        return False

# ==========================================
# 3. 状态与辅助逻辑
# ==========================================
def get_target_idx(agent_id, task_status, task_locations, my_pos):
    """
    策略核心：静态协同分工
    Agent 0 处理前3个任务 (0,1,2)
    Agent 1 处理后3个任务 (3,4,5)
    """
    my_tasks = range(3) if agent_id == 0 else range(3, 6)
    
    closest_dist = float('inf')
    closest_idx = -1
    
    for i in my_tasks:
        if task_status[i] == 1:
            dist = np.sum(np.abs(my_pos - task_locations[i]))
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i
    return closest_idx

def make_state(obs, agent_id):
    """状态：相对目标的方位 (dx, dy)"""
    my_pos = obs["robots"][agent_id]
    target_idx = get_target_idx(agent_id, obs["task_status"], obs["tasks"], my_pos)
    
    if target_idx != -1:
        diff = obs["tasks"][target_idx] - my_pos
        # 压缩状态空间：只取符号 (-1, 0, 1)
        return (int(np.sign(diff[0])), int(np.sign(diff[1])))
    return (0, 0) # 无任务状态

def get_shaping(obs, next_obs, agent_id):
    """奖励塑造：辅助收敛"""
    p1 = obs["robots"][agent_id]
    t1_idx = get_target_idx(agent_id, obs["task_status"], obs["tasks"], p1)
    d1 = np.sum(np.abs(obs["tasks"][t1_idx] - p1)) if t1_idx != -1 else 0
    
    p2 = next_obs["robots"][agent_id]
    t2_idx = get_target_idx(agent_id, next_obs["task_status"], next_obs["tasks"], p2)
    d2 = np.sum(np.abs(next_obs["tasks"][t2_idx] - p2)) if t2_idx != -1 else 0
    
    if t1_idx != -1 and t2_idx != -1 and t1_idx == t2_idx:
        return (d1 - d2) * 0.5 # 靠近给正反馈
    return 0

# ==========================================
# 4. 主程序
# ==========================================
def main():
    print(">>> 启动室内服务机器人协同调度仿真 (Thesis Final Version)...")
    env = RobotGridEnv(num_tasks=6)
    agents = [QLearningAgent(0, 5), QLearningAgent(1, 5)]
    
    num_episodes = 1500
    epsilon = 1.0
    epsilon_min = 0.01 # 允许微量探索
    
    history = {"reward": [], "steps": [], "wait_time": []}
    best_avg_steps = float('inf')

    # --- 训练阶段 ---
    print(f"开始训练 ({num_episodes} Episodes)...")
    start_time = time.time()
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        terminated = False
        steps = 0
        eps_reward = np.zeros(2)
        cumulative_wait = 0
        
        # 衰减 Epsilon
        epsilon = max(epsilon_min, epsilon * 0.995)
        for a in agents: a.epsilon = epsilon
        
        while not terminated and steps < 200:
            actions = []
            cumulative_wait += np.sum(obs["task_status"]) # 累积等待任务数
            
            for i, agent in enumerate(agents):
                state = make_state(obs, i)
                actions.append(agent.choose_action(state))
                
            next_obs, rewards, terminated, _, _ = env.step(actions)
            
            for i, agent in enumerate(agents):
                # 学习
                s = make_state(obs, i)
                ns = make_state(next_obs, i)
                # 组合奖励：环境奖励 + 距离引导
                r = rewards[i] + get_shaping(obs, next_obs, i)
                agent.learn(s, actions[i], r, ns)
                
            obs = next_obs
            eps_reward += rewards
            steps += 1
            
        # 记录数据
        history["reward"].append(np.sum(eps_reward))
        history["steps"].append(steps)
        history["wait_time"].append(cumulative_wait / 6.0) # 平均每个任务等待时长
        
        # 日志与保存最佳
        if (episode+1) % 100 == 0:
            avg_steps = np.mean(history["steps"][-50:])
            avg_rew = np.mean(history["reward"][-50:])
            print(f"Ep {episode+1}: Steps={avg_steps:.1f}, Reward={avg_rew:.1f}, Epsilon={epsilon:.2f}")
            
            if avg_steps < best_avg_steps:
                best_avg_steps = avg_steps
                for a in agents: a.save(is_best=True)
                print(f"  [New Record] Models saved. Best Steps: {best_avg_steps:.1f}")

    print(f"训练完成，耗时 {time.time()-start_time:.1f}s")

    # --- 绘图 (最终证据) ---
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 4))
        
        # 收敛性
        plt.subplot(1, 3, 1)
        plt.plot(np.convolve(history["reward"], np.ones(50)/50, mode='valid'))
        plt.title("Convergence (Reward)")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True, alpha=0.3)
        
        # 吞吐量 (转化为 Steps, 越低吞吐量越高)
        plt.subplot(1, 3, 2)
        plt.plot(np.convolve(history["steps"], np.ones(50)/50, mode='valid'), color='orange')
        plt.title("Throughput Optimization (Steps)")
        plt.xlabel("Episode")
        plt.ylabel("Steps to Complete All")
        plt.grid(True, alpha=0.3)
        
        # 等待时间
        plt.subplot(1, 3, 3)
        plt.plot(np.convolve(history["wait_time"], np.ones(50)/50, mode='valid'), color='green')
        plt.title("Waiting Time Optimization")
        plt.xlabel("Episode")
        plt.ylabel("Avg Waiting Time")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("paper_results.png")
        print("\n>>> 最终图表已生成: paper_results.png")
        print(">>> 请查看图表，这次的 Wait Time 和 Steps 应该会有显著下降趋势。")
    except:
        pass

if __name__ == "__main__":
    main()