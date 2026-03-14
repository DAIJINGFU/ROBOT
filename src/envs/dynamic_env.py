import gymnasium as gym
from gymnasium import spaces
import numpy as np

# A more complex grid layout representing a floor plan
# H: Hallway/Corridor (Open)
# W: Wall
# D: Door (Dynamic - Can close)
# C: Charging Station
# P: Pickup Point (Task Source)
# O: Office/Workstation Step
GRID_LAYOUT = [
    "WWWWWWWWWWWW",
    "WC...D....CW",
    "W....W.....W",
    "W.O..W..O..W",
    "W....D.....W",
    "WWWW...WWWWW",
    "....H.H.....",
    "WWWW...WWWWW",
    "W....D.....W",
    "W.O..W..O..W",
    "W....W.....W",
    "WP...D....PW",
    "WWWWWWWWWWWW"
]

class Task:
    def __init__(self, t_id, pos, task_type, priority, deadline, creation_time):
        self.id = t_id
        self.pos = np.array(pos)
        self.type = task_type # 0: Delivery, 1: Inspection, 2: Guide
        self.priority = priority # 1: Low, 2: High, 3: Emergency
        self.deadline = deadline
        self.creation_time = creation_time
        self.service_duration = np.random.randint(2, 5) # Steps to complete
        self.progress = 0
        self.assigned_robot = -1

class DynamicRobotGridEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.grid_size = 13 # Based on layout
        self.num_robots = 2
        self.render_mode = render_mode
        self.max_steps = 500
        
        # Parse Layout
        self.layout = np.array([list(row) for row in GRID_LAYOUT])
        self.walls = np.argwhere(self.layout == 'W')
        self.doors = np.argwhere(self.layout == 'D')
        self.chargers = np.argwhere(self.layout == 'C')
        self.pickups = np.argwhere(self.layout == 'P')
        self.offices = np.argwhere(self.layout == 'O')
        
        # Dynamic Elements
        self.door_status = np.ones(len(self.doors), dtype=int) # 1=Open, 0=Closed
        self.pedestrians = [] # List of positions
        self.tasks = []
        self.task_counter = 0
        
        # Robot Config
        self.robot_capacity = [2, 1] # Robot 0: Heavy (2 items), Robot 1: Fast (1 item)
        self.robot_speeds = [1, 2]   # Robot 0: Slow, Robot 1: Fast (Not implemented in grid yet, abstract concept)
        self.robot_battery_max = 100
        
        # Observation Space (Simplified for RL input)
        # We need a fixed size observation for standard RL, but tasks are dynamic.
        # We will expose a "Sensor Map" approach or a fixed list of "Nearest Tasks".
        self.observation_space = spaces.Dict({
            "robots": spaces.Box(0, self.grid_size, shape=(self.num_robots, 2), dtype=np.int32),
            "battery": spaces.Box(0, 100, shape=(self.num_robots,), dtype=np.int32),
            # "sensor_map": spaces.Box(0, 3, shape=(self.grid_size, self.grid_size), dtype=np.int32) # Too large for simple Q-Learning
            # Instead, we provide list of active task locations (padded)
            "target_vector": spaces.Box(-1, self.grid_size, shape=(self.num_robots, 2), dtype=np.int32) # Relative vector to assigned task
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
        self._robot_locations = self.chargers[0:2].copy() if len(self.chargers) >= 2 else np.array([[1,1], [1,2]])
        self._robot_battery = np.full(self.num_robots, self.robot_battery_max, dtype=np.int32)
        self._robot_load = np.zeros(self.num_robots, dtype=np.int32) # Current load
        self._dead_robots = [False] * self.num_robots
        
        self.tasks = []
        self.task_counter = 0
        self.door_status = np.ones(len(self.doors), dtype=int)
        
        # Initial Tasks
        for _ in range(4): self._generate_task()
        
        return self._get_obs(), {}

    def _generate_task(self):
        # 30% Delivery (P -> O), 40% Inspection (O), 30% Guide (H -> O)
        r = np.random.rand()
        t_pos = None
        t_type = 0
        
        if r < 0.3:
            # Delivery: Start at Pickup
            idx = np.random.randint(len(self.pickups))
            t_pos = self.pickups[idx]
            t_type = 0
        elif r < 0.7:
            # Inspection: At Office
            idx = np.random.randint(len(self.offices))
            t_pos = self.offices[idx]
            t_type = 1
        else:
            # Guide: Random valid point
            while True:
                pos = np.random.randint(0, self.grid_size, size=2)
                if self.layout[pos[0], pos[1]] not in ['W', 'C', 'P']:
                    t_pos = pos
                    break
            t_type = 2
            
        priority = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
        deadline = self.current_step + np.random.randint(50, 200)
        
        new_task = Task(self.task_counter, t_pos, t_type, priority, deadline, self.current_step)
        self.tasks.append(new_task)
        self.task_counter += 1

    def _get_obs(self):
        # Flatten observation for simple agents
        return {
            "robots": self._robot_locations.copy(),
            "battery": self._robot_battery.copy(),
            "layout": self.layout.copy(), # Static
            "doors": self.door_status.copy(),
            "tasks": self.tasks # Raw object list for heuristic usage
        }
        
    def step(self, action):
        self.current_step += 1
        rewards = np.zeros(self.num_robots)
        terminated = False
        truncated = False
        
        # 1. Dynamic Events (Disturbances)
        # Random Door Toggle (Low prob)
        if np.random.rand() < 0.05:
            d_idx = np.random.randint(len(self.doors))
            self.door_status[d_idx] = 1 - self.door_status[d_idx]
            
        # Pedestrians (Simple: Random blockage)
        # TODO: Implement actual moving pedestrians
        pass
        
        # Task Arrival (Poisson-ish)
        if np.random.rand() < 0.1 and len(self.tasks) < 10:
             self._generate_task()
        
        # 2. Robot Movement
        for i in range(self.num_robots):
            if self._dead_robots[i]: continue
            
            # Action Execution
            move_vec = self._action_to_direction[action[i]]
            curr_pos = self._robot_locations[i]
            proposed_pos = curr_pos + move_vec
            
            # Collision Checks
            valid_move = True
            
            # Wall Check
            if not (0 <= proposed_pos[0] < self.grid_size and 0 <= proposed_pos[1] < self.grid_size):
                valid_move = False
            elif self.layout[proposed_pos[0], proposed_pos[1]] == 'W':
                valid_move = False
            
            # Door Check
            for d_idx, door_pos in enumerate(self.doors):
                if np.array_equal(proposed_pos, door_pos) and self.door_status[d_idx] == 0:
                    valid_move = False # Door Closed!
                    rewards[i] -= 0.5 # Penalty for hitting closed door
                    
            # Update Position
            if valid_move:
                self._robot_locations[i] = proposed_pos
                self._robot_battery[i] -= 1 # Move Cost
            else:
                rewards[i] -= 1.0 # Bump Penalty
                
            # Charging Logic
            at_charger = False
            for c_pos in self.chargers:
                if np.array_equal(self._robot_locations[i], c_pos):
                    at_charger = True
                    break
            
            if at_charger and action[i] == 4: # Stay action at charger
                self._robot_battery[i] = min(100, self._robot_battery[i] + 10)
                
            if self._robot_battery[i] <= 0:
                self._dead_robots[i] = True
                rewards[i] -= 100.0 # Huge penalty
        
        # 3. Task Processing
        tasks_to_remove = []
        for t in self.tasks:
            # Check Deadline
            if self.current_step > t.deadline:
                rewards[:] -= (t.priority * 2.0) # Global Penalty for missed deadline
                tasks_to_remove.append(t)
                continue
                
            # Check Completion
            for r_idx in range(self.num_robots):
                if self._dead_robots[r_idx]: continue
                
                # If robot is at task location
                if np.array_equal(self._robot_locations[r_idx], t.pos):
                    # Service logic
                    t.progress += 1
                    if t.progress >= t.service_duration:
                        # Task Done
                        reward_val = 10.0 * t.priority
                        rewards[r_idx] += reward_val
                        if t not in tasks_to_remove: tasks_to_remove.append(t)
                        
        for t in tasks_to_remove:
            if t in self.tasks: self.tasks.remove(t)
            
        # 4. Global Penalty (Waiting Time)
        # Sum of priorities of active tasks
        congestion_penalty = sum([t.priority for t in self.tasks]) * 0.1
        rewards -= congestion_penalty
        
        if self.current_step >= self.max_steps:
            truncated = True
            
        return self._get_obs(), rewards, terminated, truncated, {}
