import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os
import heapq # For A* pathfinding

# 复用 RobotGridEnv 保证自包含
class RobotGridEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size=12, num_robots=2, num_tasks=6, render_mode=None, fixed_tasks=True):
        self.grid_size = grid_size
        self.num_robots = num_robots
        self.num_tasks = num_tasks  
        self.render_mode = render_mode
        self.fixed_tasks = fixed_tasks
        
        self.max_battery = 100
        self.battery_decay = 1
        self.min_battery_threshold = 20 
        
        # 1. 充电站 (左下角, 右上角)
        self.charging_stations = np.array([[0, 0], [grid_size-1, grid_size-1]], dtype=np.int32)
        
        # 2. 障碍物 (H形走廊)
        self.obstacles = []
        for y in range(6): self.obstacles.append([4, y]) # Wall Left
        for y in range(6, 12): self.obstacles.append([7, y]) # Wall Right
        self.obstacles = np.array(self.obstacles, dtype=np.int32)
        
        self.observation_space = spaces.Dict({
            "robots": spaces.Box(0, grid_size, shape=(num_robots, 2), dtype=np.int32),
            "battery": spaces.Box(0, self.max_battery, shape=(num_robots,), dtype=np.int32), 
            "tasks": spaces.Box(0, grid_size, shape=(num_tasks, 2), dtype=np.int32),
            "task_status": spaces.MultiBinary(num_tasks) 
        })

        self.action_space = spaces.MultiDiscrete([5] * num_robots)
        
        self._action_to_direction = {
            0: np.array([0, 1]), 1: np.array([0, -1]),
            2: np.array([1, 0]), 3: np.array([-1, 0]),
            4: np.array([0, 0]), 
        }
        
        if fixed_tasks:
            self._fixed_task_locations = np.array([
                [1, 1], [1, 8], [5, 2], 
                [8, 1], [8, 8], [6, 7]
            ][:num_tasks])
        else:
            self._fixed_task_locations = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._robot_locations = np.array([[5, 5], [6, 6]], dtype=np.int32)
        self._robot_battery = np.full(self.num_robots, self.max_battery, dtype=np.int32)
        self._dead_robots = [False] * self.num_robots 
        
        if self.fixed_tasks and self._fixed_task_locations is not None:
             self._task_locations = self._fixed_task_locations.copy()
        else:
             self._task_locations = self.np_random.integers(0, self.grid_size, size=(self.num_tasks, 2), dtype=np.int32)

        self._task_status = np.ones(self.num_tasks, dtype=np.int8)
        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "robots": self._robot_locations.copy(),
            "battery": self._robot_battery.copy(), 
            "tasks": self._task_locations.copy(),
            "task_status": self._task_status.copy(),
            "charging_stations": self.charging_stations.copy(),
            "obstacles": self.obstacles.copy()
        }

    def step(self, action):
        rewards = np.zeros(self.num_robots)
        terminated = False
        truncated = False
        
        for i in range(self.num_robots):
            if self._dead_robots[i]: continue 
            
            move_vec = self._action_to_direction[action[i]]
            current_pos = self._robot_locations[i]
            is_staying = (action[i] == 4)
            at_charger = any(np.array_equal(current_pos, cs) for cs in self.charging_stations)
            
            if is_staying and at_charger:
                if self._robot_battery[i] < self.max_battery:
                    inc = 20
                    self._robot_battery[i] = min(self.max_battery, self._robot_battery[i] + inc)
                    rewards[i] += 5.0 # Higher reward for charging correctly
            else:
                proposed_pos = current_pos + move_vec
                
                # Check walls
                hit = False
                if not (0 <= proposed_pos[0] < self.grid_size and 0 <= proposed_pos[1] < self.grid_size):
                    hit = True
                else:
                    for obs in self.obstacles:
                        if np.array_equal(proposed_pos, obs):
                            hit = True
                            break
                
                if hit:
                    rewards[i] -= 2.0 # Higher penalty for bumping
                else:
                    self._robot_locations[i] = proposed_pos.astype(np.int32)
                
                if not is_staying:
                    self._robot_battery[i] -= self.battery_decay
            
            if self._robot_battery[i] <= 0:
                self._robot_battery[i] = 0
                self._dead_robots[i] = True
                rewards[i] -= 50.0 
        
        active_tasks = np.sum(self._task_status)
        for t_idx in range(self.num_tasks):
            if self._task_status[t_idx] == 1:
                task_pos = self._task_locations[t_idx]
                for r_idx in range(self.num_robots):
                    if not self._dead_robots[r_idx] and np.array_equal(self._robot_locations[r_idx], task_pos):
                        self._task_status[t_idx] = 0 
                        rewards[r_idx] += 30.0 
                        break 
        
        rewards -= (0.1 + 0.05 * active_tasks) 
        
        if active_tasks == 0 or all(self._dead_robots):
            terminated = True
            
        return self._get_obs(), rewards, terminated, truncated, {}

# ==========================================
# 2. 算法与寻路 (BFS + Q-Learning)
# ==========================================
class QLearningAgent:
    def __init__(self, agent_id, action_space_size, epsilon=0.1):
        self.agent_id = agent_id
        self.lr = 0.1
        self.gamma = 0.95
        self.epsilon = epsilon
        self.q_table = {}

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(5)
        return np.argmax(self.q_table.get(state, np.zeros(5)))

    def learn(self, state, action, reward, next_state):
        current_q = self.q_table.get(state, np.zeros(5))
        next_q = self.q_table.get(next_state, np.zeros(5))
        current_q[action] += self.lr * (reward + self.gamma * np.max(next_q) - current_q[action])
        self.q_table[state] = current_q
    
    def save(self, name):
        with open(name, 'wb') as f: pickle.dump(self.q_table, f)

def get_path_next_step(start_pos, goal_pos, grid_size, obstacles):
    """BFS: Find shortest next step to goal avoiding obstacles."""
    if np.array_equal(start_pos, goal_pos): return 0 # Stay
    
    start_tuple = tuple(start_pos)
    goal_tuple = tuple(goal_pos)
    obstacle_set = set(tuple(o) for o in obstacles)
    
    queue = [(start_tuple, [])] # (current, path)
    visited = {start_tuple}
    
    parent_map = {} # curr -> prev
    
    # Simple BFS to find goal
    found = False
    front = [start_tuple]
    
    # Proper BFS for reconstruction
    q = [(start_tuple)]
    came_from = {start_tuple: None}
    
    while q:
        current = q.pop(0)
        if current == goal_tuple:
            found = True
            break
        
        cx, cy = current
        for dx, dy, action in [(0,1,0), (0,-1,1), (1,0,2), (-1,0,3)]: # Right, Left, Down, Up
            nx, ny = cx+dx, cy+dy
            next_node = (nx, ny)
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                if next_node not in obstacle_set and next_node not in came_from:
                    came_from[next_node] = (current, action) # Store action to get here
                    q.append(next_node)
    
    if not found: return 4 # Stay if unreachable
    
    # Backtrack to find first move
    curr = goal_tuple
    first_action = 4
    while curr != start_tuple:
        prev, action = came_from[curr]
        if prev == start_tuple:
            first_action = action
            break
        curr = prev
    return first_action

def make_state(obs, agent_id, env):
    """
    状态 = (BatteryMode, OptimalAction)
    Agent不需要探索迷宫，只需要决定【做任务】还是【去充电】。
    OptimalAction由BFS计算，告诉Agent怎么走最快。
    Agent只需要学习：当没电时，听从“去充电”的建议；有电时，听从“去任务”的建议。
    """
    my_pos = obs["robots"][agent_id]
    battery = obs["battery"][agent_id]
    task_locations = obs["tasks"]
    task_status = obs["task_status"]
    charging_stations = obs.get("charging_stations")
    
    # 模式判定
    # 0 = Task Mode, 1 = Charge Mode
    # 这里的阈值可以交给Agent去适应，但在离散状态里我们先硬编码
    mode = 1 if battery < 30 else 0 
    
    target_pos = None
    
    if mode == 1: # Charge
        # Find nearest charger
        best_dist = 999
        for cs in charging_stations:
            # Heuristic distance sufficient for selection
            d = abs(my_pos[0]-cs[0]) + abs(my_pos[1]-cs[1]) 
            if d < best_dist:
                best_dist = d
                target_pos = cs
    else: # Task
        # Find nearest allocated task
        best_dist = 999
        start = 0 if agent_id == 0 else 3
        end = 3 if agent_id == 0 else 6
        
        # Check own tasks
        for i in range(start, min(end, len(task_status))):
            if task_status[i] == 1:
                d = abs(my_pos[0]-task_locations[i][0]) + abs(my_pos[1]-task_locations[i][1])
                if d < best_dist:
                    best_dist = d
                    target_pos = task_locations[i]
        
        # If no own tasks, help others
        if target_pos is None:
            for i in range(len(task_status)):
                if task_status[i] == 1:
                     # Simple logic
                     target_pos = task_locations[i]
                     break
                     
    if target_pos is None:
        rec_action = 4 # Stay
    else:
        rec_action = get_path_next_step(my_pos, target_pos, env.grid_size, env.obstacles)
        
    # Agent 状态 = (模式, 推荐动作)
    # 这样 Q-Table Size = 2 * 5 = 10. 极小！
    # 收敛极其快。
    return (mode, rec_action)

def main():
    print("初始化增强环境 (含障碍物与电池)...")
    env = RobotGridEnv(grid_size=12, num_robots=2, num_tasks=6)
    
    agents = [QLearningAgent(0, 5, 0.5), QLearningAgent(1, 5, 0.5)]
    history = {"r": [], "s": [], "b": []}
    
    start_time = time.time()
    
    # 因为引入了 BFS 做底层导航，其实不需要训练 2000 次
    # Agent 只需要学会：低电量时跟随去充电的导航，高电量时跟随去任务的导航
    # 甚至不需要 RL？
    # 不，RL 的作用是学习【何时切换任务】以及【避免拥堵】(虽然这里简化了拥堵)
    # 以及学习【阈值】（虽然我们硬编码了 mode）
    
    # 为了演示学习过程，我们保留 RL
    num_episodes = 500 
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        term = False
        steps = 0
        ep_r = np.zeros(2)
        
        # Epsilon decay
        for ag in agents: ag.epsilon *= 0.95
        
        while not term and steps < 200:
            actions = []
            states = []
            for i, ag in enumerate(agents):
                s = make_state(obs, i, env)
                states.append(s)
                # 能够覆盖 BFS 建议
                # 如果 epsilon 高，可能会乱走撞墙，然后受到惩罚
                action = ag.choose_action(s)
                
                # HACK: 如果 Action != rec_action，大概率是乱走
                # 为了加速演示效果，我们可以把 Rec_Action 作为一个 Feature
                # 但在这里，我们让 Agent 自己选
                actions.append(action)
            
            next_obs, rewards, term, trunc, _ = env.step(actions)
            
            for i, ag in enumerate(agents):
                ns = make_state(next_obs, i, env)
                ag.learn(states[i], actions[i], rewards[i], ns)
                
            obs = next_obs
            ep_r += rewards
            steps += 1
            
        history["r"].append(np.sum(ep_r))
        history["s"].append(steps)
        history["b"].append(np.mean(obs["battery"]))
        
        if (episode+1) % 50 == 0:
            print(f"Ep {episode+1}: Rew={np.mean(history['r'][-50:]):.1f}, Steps={np.mean(history['s'][-50:]):.1f}, Bat={np.mean(history['b'][-50:]):.1f}")

    print(f"训练完成. 耗时 {time.time()-start_time:.1f}s")
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1,3,1); plt.plot(history["r"]); plt.title("Rewards")
    plt.subplot(1,3,2); plt.plot(history["s"]); plt.title("Steps")
    plt.subplot(1,3,3); plt.plot(history["b"]); plt.title("Final Battery")
    plt.savefig("paper_results_v2.png")
    print("图表已生: paper_results_v2.png")

if __name__ == "__main__":
    main()
