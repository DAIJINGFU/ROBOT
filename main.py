import numpy as np
import time
from src.envs.robot_env import RobotGridEnv
from src.agents.q_learning_agent import QLearningAgent

def get_dist_to_nearest(obs, agent_id):
    """计算到最近任务的曼哈顿距离 (遵循分工逻辑)"""
    my_pos = obs["robots"][agent_id]
    task_locations = obs["tasks"]
    task_status = obs["task_status"]
    
    min_dist = float('inf')
    found = False
    
    # 优先找自己的任务
    for i, status in enumerate(task_status):
        if status == 1 and i % 2 == agent_id:
            dist = np.sum(np.abs(my_pos - task_locations[i]))
            if dist < min_dist:
                min_dist = dist
                found = True
                
    # 如果没有自己的任务，找任意任务
    if not found:
        for i, status in enumerate(task_status):
            if status == 1:
                dist = np.sum(np.abs(my_pos - task_locations[i]))
                if dist < min_dist:
                    min_dist = dist
                    found = True
                    
    return min_dist if found else 0

def make_state_hashable(obs, agent_id):
    """
    将观察转换为状态元组。
    优化：加入【最近任务的相对方向】，引导机器人向任务移动。
    State = (my_x, my_y, dir_x, dir_y)
    """
    my_pos = obs["robots"][agent_id]
    task_locations = obs["tasks"]
    task_status = obs["task_status"]
    
    # 1. 寻找最近的未完成任务
    # 策略优化：静态分工 (Static Partitioning) 以避免抢夺同一个任务
    # 机器人0 只关注任务 0, 2, 4... | 机器人1 只关注任务 1, 3, 5...
    closest_dist = float('inf')
    closest_idx = -1
    
    for i, status in enumerate(task_status):
        if status == 1: # Active
            # 分工过滤：如果这个任务不归我管，就假装看不见
            if i % 2 != agent_id:
                continue
                
            dist = np.sum(np.abs(my_pos - task_locations[i]))
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i
    
    # 如果自己的任务都做完了，但场上还有别的任务（别人的），能否帮忙？
    # 简单起见：先只管自己的。如果自己的做完了，就停下（return 0,0）或者去最近的任意任务
    if closest_idx == -1:
        # Fallback: 帮忙处理任意剩余任务
        for i, status in enumerate(task_status):
            if status == 1:
                dist = np.sum(np.abs(my_pos - task_locations[i]))
                if dist < closest_dist:
                    closest_dist = dist
                    closest_idx = i
    
    # 2. 计算相对方向
    if closest_idx != -1:
        target_pos = task_locations[closest_idx]
        dx = target_pos[0] - my_pos[0]
        dy = target_pos[1] - my_pos[1]
        
        dir_x = int(np.sign(dx)) # -1, 0, 1
        dir_y = int(np.sign(dy)) # -1, 0, 1
        
        # 极端简化：只使用相对方向作为状态。
        # 这样状态空间仅有 3*3=9 个状态，训练将极其迅速。
        # 适用于无障碍物的 Grid World。
        return (dir_x, dir_y)
    else:
        # 所有任务都完成了
        return (0, 0)

def main():
    print("初始化环境...")
    # 创建环境：10x10网格，2个机器人，5个任务，固定任务位置以便训练
    env = RobotGridEnv(grid_size=10, num_robots=2, num_tasks=5, render_mode=None, fixed_tasks=True)
    
    # 创建智能体
    agents = [
        QLearningAgent(agent_id=0, action_space_size=5, epsilon=1.0),
        QLearningAgent(agent_id=1, action_space_size=5, epsilon=1.0)
    ]
    
    num_episodes = 2000  # 恢复为 2000 回合以保证收敛效果
    epsilon_decay = 0.995
    min_epsilon = 0.10 # 保持 10% 的探索率，防止陷入死循环
    
    print(f"开始运行 {num_episodes} 个回合...")
    
    # 记录数据用于绘图 - 对应任务书的关键指标
    history = {
        "rewards": [],
        "steps": [],
        "throughput": [],
        "waiting_time": [] # 平均等待时间
    }

    start_time = time.time()
    best_reward = -float('inf')
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        terminated = False
        step_count = 0
        total_rewards = np.zeros(len(agents))
        
        # 统计本回合所有任务的累积等待时间
        cumulative_waiting_time = 0 
        
        # 衰减 Epsilon
        for agent in agents:
            agent.epsilon = max(min_epsilon, agent.epsilon * epsilon_decay)
            
        while not terminated and step_count < 200: 
            actions = []
            
            # 记录当前未完成的任务数作为这一步的等待惩罚
            active_tasks_count = np.sum(obs["task_status"])
            cumulative_waiting_time += active_tasks_count
            
            # 1. 每个智能体选择动作
            for i, agent in enumerate(agents):
                state = make_state_hashable(obs, i)
                action = agent.choose_action(state)
                actions.append(action)
            
            # 2. 环境执行动作
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            
            # 3. 智能体学习 (Learn)
            for i, agent in enumerate(agents):
                # 计算“奖励塑造”(Reward Shaping)
                # 注意：如果刚刚完成了任务（获得大额奖励），目标会切换，距离会突变。
                # 此时不应计算距离差惩罚，否则会抵消完成任务的奖励。
                if rewards[i] >= 5.0: # 假设完成任务奖励是 10.0，这里设阈值判断
                    shaping = 0
                else:
                    old_dist = get_dist_to_nearest(obs, i)
                    new_dist = get_dist_to_nearest(next_obs, i)
                    shaping = (old_dist - new_dist) * 0.5 
                
                # 组合总奖励
                total_reward = rewards[i] + shaping
                
                state = make_state_hashable(obs, i)
                next_state = make_state_hashable(next_obs, i)
                agent.learn(state, actions[i], total_reward, next_state)
                
            obs = next_obs
            total_rewards += rewards
            step_count += 1
        
        # 记录本回合数据
        # 吞吐量 (Throughput): 完成任务数(5) / 整个Episode耗时
        # 平均等待时间 (Avg Waiting Time) = 总累积等待时间 / 任务总数(5)
        # 注意: 如果 step_count 很大，吞吐量就很小；等待时间很大。
        
        # 修正逻辑：只有当 terminated=True (所有任务完成) 时，才记录有效的吞吐量
        # 如果超时(step_count == 200)，吞吐量应该惩罚性计算 (例如完成数/200)
        tasks_completed = 5 - np.sum(obs["task_status"])
        throughput = tasks_completed / step_count if step_count > 0 else 0
        avg_wait = cumulative_waiting_time / 5.0
        
        history["rewards"].append(np.sum(total_rewards))
        history["steps"].append(step_count) # 修复：确保记录步数，避免 Nan 错误
        history["throughput"].append(throughput)
        history["waiting_time"].append(avg_wait)
        
        if (episode + 1) % 50 == 0:
            # 格式化优化：Steps比小数点的吞吐量更直观
            current_avg_reward = np.mean(history['rewards'][-50:])
            print(f"Episode {episode+1}: Steps={np.mean(history['steps'][-50:]):.1f}, Reward={current_avg_reward:.1f}, WaitTime={np.mean(history['waiting_time'][-50:]):.1f}")
            
            # 保存最佳模型
            if current_avg_reward > best_reward:
                best_reward = current_avg_reward
                for i, agent in enumerate(agents):
                    agent.save(f"agent_{i}_best.pkl")
                print(f"  >>> 新纪录！最佳模型已保存 (AvgReward={best_reward:.2f})")

    print(f"训练结束，总耗时 {time.time() - start_time:.2f}s")
    
    # 保存最终模型
    for i, agent in enumerate(agents):
        agent.save(f"agent_{i}_final.pkl")
    print("模型已保存。")

    # 绘制符合任务书要求的论文插图
    try:
        import matplotlib.pyplot as plt
        # 设置字体，防止中文乱码（虽然这里先用英文标签）
        plt.rcParams['axes.unicode_minus'] = False 

        plt.figure(figsize=(15, 5))
        
        # 图1: 收敛曲线 (证明方法有效)
        plt.subplot(1, 3, 1)
        plt.plot(history["rewards"])
        plt.title("Convergence Analysis (Reward)")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True, alpha=0.3)
        
        # 图2: 吞吐量优化 (任务书核心指标1)
        # 使用移动平均线让曲线更好看
        window = 20
        throughput_smooth = np.convolve(history["throughput"], np.ones(window)/window, mode='valid')
        plt.subplot(1, 3, 2)
        plt.plot(throughput_smooth, color='orange')
        plt.title("System Throughput Optimization")
        plt.xlabel("Episode")
        plt.ylabel("Tasks per Step")
        plt.grid(True, alpha=0.3)
        
        # 图3: 等待时间优化 (任务书核心指标2)
        wait_smooth = np.convolve(history["waiting_time"], np.ones(window)/window, mode='valid')
        plt.subplot(1, 3, 3)
        plt.plot(wait_smooth, color='green')
        plt.title("Avg Task Waiting Time Optimization")
        plt.xlabel("Episode")
        plt.ylabel("Time Steps")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("paper_results.png", dpi=300) # 高分辨率保存
        print("论文与答辩专用图表已保存为 paper_results.png")
        
    except ImportError:
        print("matplotlib 未安装，跳过绘图。")

if __name__ == "__main__":
    main()
