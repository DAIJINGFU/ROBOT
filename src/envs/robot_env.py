import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RobotGridEnv(gym.Env):
    """
    一个增强版的室内服务机器人环境 (Grid World v2).
    
    特性 (符合任务书要求的“场景建模”):
    1. 障碍物 (Obstacles): 模拟墙壁和家具。
    2. 充电站 (Charging Stations): 机器人需要充电。
    3. 电池动力学 (Battery Dynamics): 移动消耗电量，低电量需返回充电。
    4. 动态任务 (即使是固定任务列表，也会考虑任务状态)。
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size=12, num_robots=2, num_tasks=6, render_mode=None, fixed_tasks=True):
        self.grid_size = grid_size
        self.num_robots = num_robots
        self.num_tasks = num_tasks  
        self.render_mode = render_mode
        self.fixed_tasks = fixed_tasks
        
        # 电池参数
        self.max_battery = 100
        self.battery_decay = 1
        self.min_battery_threshold = 20 # 低电量阈值
        
        # 充电站位置 (固定)
        # 设有两个充电桩：左下角(0,0) 和 右上角(11,11)
        self.charging_stations = np.array([[0, 0], [grid_size-1, grid_size-1]], dtype=np.int32)
        
        # 障碍物位置 (模拟房间墙壁)
        # 设计为一个简单的 "H" 型走廊 或 两个房间
        self.obstacles = []
        # 墙1: x=4, y=0~5 (左边房间的墙)
        for y in range(6): self.obstacles.append([4, y])
        # 墙2: x=7, y=6~11 (右边房间的墙)
        for y in range(6, 12): self.obstacles.append([7, y])
        self.obstacles = np.array(self.obstacles, dtype=np.int32)
        
        # 观测空间
        self.observation_space = spaces.Dict({
            "robots": spaces.Box(0, grid_size, shape=(num_robots, 2), dtype=np.int32),
            "battery": spaces.Box(0, self.max_battery, shape=(num_robots,), dtype=np.int32), # 新增：电池状态
            "tasks": spaces.Box(0, grid_size, shape=(num_tasks, 2), dtype=np.int32),
            "task_status": spaces.MultiBinary(num_tasks) 
        })

        # 动作: 0=右, 1=左, 2=下, 3=上, 4=停(充电/待机)
        self.action_space = spaces.MultiDiscrete([5] * num_robots)
        
        self._action_to_direction = {
            0: np.array([0, 1]), 1: np.array([0, -1]),
            2: np.array([1, 0]), 3: np.array([-1, 0]),
            4: np.array([0, 0]), 
        }
        
        # 任务分布：为了体现协同，任务均匀分布
        if fixed_tasks:
            # 确保任务不在障碍物上
            self._fixed_task_locations = np.array([
                [1, 1], [1, 8], [5, 2], # 左区
                [8, 1], [8, 8], [6, 7]  # 右区
            ][:num_tasks])
        else:
            self._fixed_task_locations = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 机器人出生在中间走廊 (5,5) 和 (6,6)，避开墙壁
        self._robot_locations = np.array([[5, 5], [6, 6]], dtype=np.int32)
        
        # 初始满电
        self._robot_battery = np.full(self.num_robots, self.max_battery, dtype=np.int32)
        
        # 记录是否耗尽电量
        self._dead_robots = [False] * self.num_robots 
        
        if self.fixed_tasks and self._fixed_task_locations is not None:
             self._task_locations = self._fixed_task_locations.copy()
        else:
             # 随机生成任务，需避开障碍物
             locs = []
             for _ in range(self.num_tasks):
                 while True:
                     pos = self.np_random.integers(0, self.grid_size, size=2)
                     # 检查是否撞墙
                     is_obstacle = False
                     for obs in self.obstacles:
                         if np.array_equal(pos, obs):
                             is_obstacle = True
                             break
                     if not is_obstacle:
                         locs.append(pos)
                         break
             self._task_locations = np.array(locs)
             
        self._task_status = np.ones(self.num_tasks, dtype=np.int8)
        
        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "robots": self._robot_locations.copy(),
            "battery": self._robot_battery.copy(), # 返回电池
            "tasks": self._task_locations.copy(),
            "task_status": self._task_status.copy(),
            "charging_stations": self.charging_stations.copy(),
            "obstacles": self.obstacles.copy()
        }

    def step(self, action):
        rewards = np.zeros(self.num_robots)
        terminated = False
        truncated = False
        
        # 1. 移动与碰撞检测
        for i in range(self.num_robots):
            if self._dead_robots[i]:
                continue # 没电了动不了
                
            move_vec = self._action_to_direction[action[i]]
            
            # 如果是"停"(4) 且 在充电桩上 -> 充电
            current_pos = self._robot_locations[i]
            is_staying = (action[i] == 4)
            at_charger = False
            for cs in self.charging_stations:
                if np.array_equal(current_pos, cs):
                    at_charger = True
                    break
            
            if is_staying and at_charger:
                # 只有当电量不满时才充电，避免赖在充电桩刷分
                if self._robot_battery[i] < self.max_battery:
                    self._robot_battery[i] = min(self.max_battery, self._robot_battery[i] + 20) # 充电速度
                    rewards[i] += 2.0 # 奖励有效充电行为
            else:
                # 尝试移动
                proposed_pos = current_pos + move_vec
                
                # 边界检查
                if not (0 <= proposed_pos[0] < self.grid_size and 0 <= proposed_pos[1] < self.grid_size):
                    rewards[i] -= 1.0 # 撞墙惩罚
                    proposed_pos = current_pos # 保持不动
                
                # 障碍物检查
                hit_wall = False
                for obs in self.obstacles:
                    if np.array_equal(proposed_pos, obs):
                        hit_wall = True
                        break
                
                if hit_wall:
                    rewards[i] -= 1.0 # 撞障碍物惩罚
                    proposed_pos = current_pos # 保持不动
                
                self._robot_locations[i] = proposed_pos.astype(np.int32)
                
                # 消耗电量
                if not is_staying:
                    self._robot_battery[i] -= self.battery_decay
            
            # 电量耗尽检查
            if self._robot_battery[i] <= 0:
                self._robot_battery[i] = 0
                self._dead_robots[i] = True
                rewards[i] -= 50.0 # 严重惩罚：死机
        
        # 2. 完成任务
        active_tasks = np.sum(self._task_status)
        for t_idx in range(self.num_tasks):
            if self._task_status[t_idx] == 1:
                task_pos = self._task_locations[t_idx]
                for r_idx in range(self.num_robots):
                    if not self._dead_robots[r_idx] and np.array_equal(self._robot_locations[r_idx], task_pos):
                        self._task_status[t_idx] = 0 
                        rewards[r_idx] += 30.0 # 完成任务
                        break 
        
        # 3. 动态惩罚 (Wait Time Penalty)
        rewards -= (0.1 + 0.05 * active_tasks) 
        
        # 全体阵亡 -> 结束
        all_dead = all(self._dead_robots)
        # 任务全清 -> 结束
        all_tasks_done = (np.sum(self._task_status) == 0)
        
        if all_tasks_done or all_dead:
            terminated = True
            
        return self._get_obs(), rewards, terminated, truncated, {}
