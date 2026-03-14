import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os
import heapq # For A* pathfinding

# Map Layout Constants
MAP_WIDTH = 13
MAP_HEIGHT = 13
GRID_LAYOUT = [
    "WWWWWWWWWWWWW",
    "WC...D....CW.",
    "W....W.....W.",
    "W.O..W..O..W.",
    "W....D.....W.",
    "WWWW...WWWWW.",
    "....H.H......",
    "WWWW...WWWWW.",
    "W....D.....W.",
    "W.O..W..O..W.",
    "W....W.....W.",
    "WP...D....PW.",
    "WWWWWWWWWWWWW"
]

class Task:
    def __init__(self, t_id, pos, task_type, priority, deadline, creation_time):
        self.id = t_id
        self.pos = np.array(pos)
        self.type = task_type # 0: Delivery, 1: Inspection, 2: Guide
        self.priority = priority # 1: Low, 2: High, 3: Emergency
        self.deadline = deadline
        self.creation_time = creation_time
        self.service_duration = 3 # Fixed for simplicity
        self.progress = 0
        self.assigned_robot = -1

class DynamicRobotGridEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.grid_size = 13 # Based on layout
        self.num_robots = 2
        self.render_mode = render_mode
        self.max_steps = 300
        
        # Parse Layout
        self.layout_grid = np.array([list(row) for row in GRID_LAYOUT])
        
        self.walls = []
        self.doors = []
        self.chargers = []
        self.pickups = []
        self.offices = []
        
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = self.layout_grid[r][c]
                if cell == 'W': self.walls.append([r, c])
                elif cell == 'D': self.doors.append([r, c])
                elif cell == 'C': self.chargers.append([r, c])
                elif cell == 'P': self.pickups.append([r, c])
                elif cell == 'O': self.offices.append([r, c])
        
        self.walls = np.array(self.walls)
        self.doors = np.array(self.doors)
        self.chargers = np.array(self.chargers)
        
        # Dynamic Elements
        self.door_status = np.ones(len(self.doors), dtype=int) # 1=Open, 0=Closed
        self.tasks = []
        self.task_counter = 0
        
        # Robot Config
        self.robot_battery_max = 100
        
        self.observation_space = spaces.Dict({
            "robots": spaces.Box(0, self.grid_size, shape=(self.num_robots, 2), dtype=np.int32),
            "battery": spaces.Box(0, 100, shape=(self.num_robots,), dtype=np.int32),
        })

        self.action_space = spaces.MultiDiscrete([5] * self.num_robots)
        self._action_to_direction = {
            0: np.array([0, 1]), 1: np.array([0, -1]),
            2: np.array([1, 0]), 3: np.array([-1, 0]),
            4: np.array([0, 0]), 
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Initialize Robots at Chargers
        start_pos = [[1, 1], [1, 11]] # Hardcoded for reliability
        self._robot_locations = np.array(start_pos, dtype=np.int32)
        self._robot_battery = np.full(self.num_robots, self.robot_battery_max, dtype=np.int32)
        self._dead_robots = [False] * self.num_robots
        
        self.tasks = []
        self.task_counter = 0
        self.door_status = np.ones(len(self.doors), dtype=int)
        
        # Initial Tasks
        for _ in range(4): self._generate_task()
        
        return self._get_obs(), {}

    def _generate_task(self):
        # Generate varied tasks
        r = np.random.rand()
        t_pos = None
        t_type = 0
        
        if r < 0.4: # Office Task
            idx = np.random.randint(len(self.offices))
            t_pos = self.offices[idx]
            t_type = 1
        elif r < 0.7: # Pickup Task
            idx = np.random.randint(len(self.pickups))
            t_pos = self.pickups[idx]
            t_type = 0
        else: # Random Hallway Task
            while True:
                rr = np.random.randint(1, 11)
                cc = np.random.randint(1, 11)
                if self.layout_grid[rr][cc] == '.':
                    t_pos = [rr, cc]
                    break
            t_type = 2
            
        priority = 1
        if np.random.rand() < 0.2: priority = 3 # High Priority
        
        deadline = self.current_step + 100
        
        new_task = Task(self.task_counter, t_pos, t_type, priority, deadline, self.current_step)
        self.tasks.append(new_task)
        self.task_counter += 1

    def _get_obs(self):
        return {
            "robots": self._robot_locations.copy(),
            "battery": self._robot_battery.copy(),
            "doors": self.door_status.copy(),
            "tasks": self.tasks, # Heuristic usage
            "chargers": self.chargers
        }
        
    def step(self, action):
        self.current_step += 1
        rewards = np.zeros(self.num_robots)
        terminated = False
        truncated = False
        
        # 1. Dynamic Events
        # Random Door Toggle
        if np.random.rand() < 0.02: # Occasional door switch
            d_idx = np.random.randint(len(self.doors))
            self.door_status[d_idx] = 1 - self.door_status[d_idx]
            
        # Task Arrival
        if np.random.rand() < 0.15 and len(self.tasks) < 8:
             self._generate_task()
        
        # 2. Robot Movement
        for i in range(self.num_robots):
            if self._dead_robots[i]: continue
            
            is_staying = (action[i] == 4)
            move_vec = self._action_to_direction[action[i]]
            curr_pos = self._robot_locations[i]
            
            # Check Charger
            at_charger = False
            for c_pos in self.chargers:
                if np.array_equal(curr_pos, c_pos):
                    at_charger = True
                    break
            
            if is_staying and at_charger:
                if self._robot_battery[i] < 100:
                    self._robot_battery[i] = min(100, self._robot_battery[i] + 15)
                    rewards[i] += 2.0
            else:
                proposed_pos = curr_pos + move_vec
                
                # Collision Checks
                valid_move = True
                
                # Bounds
                if not (0 <= proposed_pos[0] < self.grid_size and 0 <= proposed_pos[1] < self.grid_size):
                    valid_move = False
                # Wall
                elif self.layout_grid[proposed_pos[0], proposed_pos[1]] == 'W':
                    valid_move = False
                
                # Door Check
                door_hit = False
                for d_idx, door_pos in enumerate(self.doors):
                    if np.array_equal(proposed_pos, door_pos):
                        if self.door_status[d_idx] == 0:
                            valid_move = False # Door Closed!
                            door_hit = True
                            
                if door_hit:
                    rewards[i] -= 2.0
                elif not valid_move:
                    rewards[i] -= 1.0
                else:
                    self._robot_locations[i] = proposed_pos
                    if not is_staying:
                        self._robot_battery[i] -= 1
            
            if self._robot_battery[i] <= 0:
                self._dead_robots[i] = True
                rewards[i] -= 50.0 
        
        # 3. Task Processing
        tasks_to_remove = []
        active_prio_sum = 0
        
        for t in self.tasks:
            # Check Deadline
            if self.current_step > t.deadline:
                rewards[:] -= (t.priority * 5.0) # Global Penalty
                tasks_to_remove.append(t)
                continue
            
            active_prio_sum += t.priority
            
            # Check Completion (Any robot nearby)
            done = False
            for r_idx in range(self.num_robots):
                if self._dead_robots[r_idx]: continue
                if np.array_equal(self._robot_locations[r_idx], t.pos):
                    # Service
                    t.progress += 1
                    if t.progress >= t.service_duration:
                        done = True
                        rewards[r_idx] += (20.0 * t.priority)
                        break
            if done:
                tasks_to_remove.append(t)
                        
        for t in tasks_to_remove:
            if t in self.tasks: self.tasks.remove(t)
            
        # 4. Congestion/Wait Penalty
        rewards -= (active_prio_sum * 0.1)
        
        if self.current_step >= self.max_steps:
            truncated = True
        
        # Terminate if all dead
        if all(self._dead_robots): terminated = True
            
        return self._get_obs(), rewards, terminated, truncated, {}

# ==========================================
# BFS Helper
# ==========================================
def get_next_move_bfs(start_pos, goal_pos, grid_layout, doors, door_status):
    """Finds next move towards goal taking Walls and Closed Doors into account."""
    if np.array_equal(start_pos, goal_pos): return 4 # Stay
    
    q = [(tuple(start_pos), [])]
    visited = {tuple(start_pos)}
    
    start_node = tuple(start_pos)
    goal_node = tuple(goal_pos)
    
    # Store parent pointers to reconstruct path
    parent = {}
    
    # BFS Queue
    queue = [start_node]
    parent[start_node] = None
    
    found = False
    
    while queue:
        curr = queue.pop(0)
        if curr == goal_node:
            found = True
            break
        
        r, c = curr
        # Explore neighbors
        for dr, dc, act in [(0,1,0), (0,-1,1), (1,0,2), (-1,0,3)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < 13 and 0 <= nc < 13:
                next_node = (nr, nc)
                if next_node not in visited:
                    cell = grid_layout[nr][nc]
                    passable = True
                    
                    if cell == 'W': passable = False
                    if cell == 'D':
                        # Find door status
                        for i, d in enumerate(doors):
                            if np.array_equal(d, [nr, nc]) and door_status[i] == 0:
                                passable = False
                                break
                    
                    if passable:
                        visited.add(next_node)
                        parent[next_node] = (curr, act)
                        queue.append(next_node)
    
    if not found: return 4 # No path found, wait
    
    # Reconstruct first move
    curr = goal_node
    while curr != start_node:
        prev_node, act = parent[curr]
        if prev_node == start_node:
            return act
        curr = prev_node
    return 4

# ==========================================
# Agent & Main
# ==========================================
class SimpleRLAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        # Simplified Q-Learning: State = (Mode), Action = (Follow bfs / Charge)
        # Actually, here we can implement a Logic-Based Agent + RL for "Task Selection"
        # Task Selection is the hard part. Navigation is solved by BFS.
        self.q_table = {} # (BatteryState, HighPrioTaskCount) -> [Task1, Task2...] 
        # For this demo, we use a fixed strategy to show the environment works
        
    def get_action(self, obs, env):
        # Heuristic Policy (Baseline)
        my_pos = obs["robots"][self.agent_id]
        battery = obs["battery"][self.agent_id]
        tasks = obs["tasks"]
        doors = obs["doors"]
        chargers = obs["chargers"]
        
        # 1. Battery Critical?
        if battery < 30:
            # Go to nearest charger
            best_dist = 999
            target = None
            for c in chargers:
                d = abs(my_pos[0]-c[0]) + abs(my_pos[1]-c[1])
                if d < best_dist:
                    best_dist = d
                    target = c
            if target is not None:
                return get_next_move_bfs(my_pos, target, env.layout_grid, env.doors, doors)
        
        # 2. Urgent Task?
        # Find highest priority task
        best_score = -999
        target = None
        
        for t in tasks:
            if t.assigned_robot != -1 and t.assigned_robot != self.agent_id: continue
            
            dist = abs(my_pos[0]-t.pos[0]) + abs(my_pos[1]-t.pos[1])
            score = t.priority * 10 - dist
            
            if score > best_score:
                best_score = score
                target = t.pos
        
        if target is not None:
            return get_next_move_bfs(my_pos, target, env.layout_grid, env.doors, doors)
            
        return np.random.randint(5) # Random wander

def main():
    print("Initialize Dynamic Indoor Environment...")
    env = DynamicRobotGridEnv()
    
    agents = [SimpleRLAgent(0), SimpleRLAgent(1)]
    
    history = {"rewards": [], "tasks_done": []}
    
    print("Running Simulation (200 Episodes)...")
    for ep in range(200):
        obs, _ = env.reset()
        done = False
        total_r = np.zeros(2)
        
        while not done:
            actions = [ag.get_action(obs, env) for ag in agents]
            obs, rewards, term, trunc, _ = env.step(actions)
            total_r += rewards
            if term or trunc: done = True
            
        history["rewards"].append(np.sum(total_r))
        
        if (ep+1) % 50 == 0:
            print(f"Episode {ep+1}: Total Reward = {np.mean(history['rewards'][-50:]):.1f}")
            
    print("Testing Complete.")
    plt.plot(history["rewards"])
    plt.title("Score in Dynamic Environment")
    plt.savefig("paper_results_v3.png")

if __name__ == "__main__":
    main()
