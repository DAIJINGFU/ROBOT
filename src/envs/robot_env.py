import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class RobotGridEnv(gym.Env):
    """
    一个简单的室内服务机器人调度环境 (Grid World).
    
    多智能体场景：
    - 多个机器人 (Agents)
    - 多个任务点 (Tasks)
    - 目标：最小化任务等待时间，最大化吞吐量
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, grid_size=10, num_robots=2, num_tasks=5, render_mode=None, fixed_tasks=True):
        self.grid_size = grid_size
        self.num_robots = num_robots
        self.num_tasks = num_tasks
        self.render_mode = render_mode
        self.fixed_tasks = fixed_tasks
        
        # 如果是固定任务，提前生成好位置
        self._fixed_task_locations = np.random.randint(0, self.grid_size, size=(self.num_tasks, 2)) if fixed_tasks else None
        
        # 定义观察空间: 简单起见，观察包括所有机器人的位置 + 所有任务的位置
        # 实际MARL中应该是局部观察
        self.observation_space = spaces.Dict({
            "robots": spaces.Box(0, grid_size, shape=(num_robots, 2), dtype=np.int32),
            "tasks": spaces.Box(0, grid_size, shape=(num_tasks, 2), dtype=np.int32),
            "task_status": spaces.MultiBinary(num_tasks) # 1=active, 0=completed
        })

        # 定义动作空间: 每个机器人有5个动作 (上, 下, 左, 右, 停)
        # 使用 MultiDiscrete 来表示多个离散动作
        self.action_space = spaces.MultiDiscrete([5] * num_robots)

        self.window = None
        self.clock = None
        
        # 动作映射
        self._action_to_direction = {
            0: np.array([0, 1]),  # Right
            1: np.array([0, -1]), # Left
            2: np.array([1, 0]),  # Down (numpy coordinates)
            3: np.array([-1, 0]), # Up
            4: np.array([0, 0]),  # Stay
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 初始化机器人位置 (随机)
        self._robot_locations = self.np_random.integers(0, self.grid_size, size=(self.num_robots, 2), dtype=np.int32)
        
        # 初始化任务位置
        if self.fixed_tasks and self._fixed_task_locations is not None:
             self._task_locations = self._fixed_task_locations.copy()
        else:
             self._task_locations = self.np_random.integers(0, self.grid_size, size=(self.num_tasks, 2), dtype=np.int32)
             
        self._task_status = np.ones(self.num_tasks, dtype=np.int8) # 全都是待处理
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _get_obs(self):
        return {
            "robots": self._robot_locations,
            "tasks": self._task_locations,
            "task_status": self._task_status
        }
    
    def _get_info(self):
        return {}

    def step(self, action):
        # action 是数组 [rob1_act, rob2_act, ...]
        
        rewards = np.zeros(self.num_robots)
        terminated = False
        truncated = False
        
        # 1. 移动机器人
        for i in range(self.num_robots):
            direction = self._action_to_direction[action[i]]
            # 计算新位置并裁剪，防止出界
            new_pos = self._robot_locations[i] + direction
            new_pos = np.clip(new_pos, 0, self.grid_size - 1)
            self._robot_locations[i] = new_pos
            
        # 2. 检查任务完成情况
        tasks_completed_this_step = 0
        for t_idx in range(self.num_tasks):
            if self._task_status[t_idx] == 1: # 如果任务还未完成
                task_pos = self._task_locations[t_idx]
                # 检查是否有机器人在此位置
                for r_idx in range(self.num_robots):
                    if np.array_equal(self._robot_locations[r_idx], task_pos):
                        self._task_status[t_idx] = 0 # 标记为完成
                        rewards[r_idx] += 10.0 # 给予该机器人奖励
                        tasks_completed_this_step += 1
                        break # 一个任务只能被一个机器人领赏（简化）
        
        # 3. 计算全局奖励 (Global Reward)
        # 例如：时间惩罚
        rewards -= 0.1
        
        # 4. 检查结束条件
        if np.sum(self._task_status) == 0:
            terminated = True
            
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, rewards, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        # 这里需要 pygame，为了简化暂时只打印文本
        # 如果需要图形化，需要安装 pygame 并实现绘图逻辑
        if self.window is None and self.render_mode == "human":
             pass # print(f"Robots: {self._robot_locations}, Tasks Left: {np.sum(self._task_status)}")
