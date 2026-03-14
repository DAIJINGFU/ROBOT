# 毕业设计项目进度与规划日志

**课题名称**：室内服务机器人协同调度：无模型多智能体RL的吞吐—等待时间多目标优化
**最后更新时间**：2026-01-18

## 1. 当前状态 (Current Status)
- **阶段**：前期准备 / 环境搭建
- **已完成事项**：
  - [x] 审查现有任务书与文件，明确课题方向。
  - [x] 确定核心技术路线：多智能体强化学习 (MARL) + 调度优化。
  - [x] **初始化项目结构**：建立了 src, envs, agents 目录及 requirements.txt。

## 2. 待办事项 (Todo List)
- [ ] **确认仿真方案**：决定使用轻量级 Python 环境还是重型 3D 仿真。（目前推荐：Python + Gymnasium）
- [ ] **环境搭建**：
  - [x] **安装依赖库** (PyTorch, Gymnasium, NumPy, Matplotlib)
  - [x] 验证环境运行
- [x] **仿真环境设计 (Environment Design)**：
  - [x] 定义地图/场景 (Grid World)
  - [x] 定义状态空间 (Dict Space)
  - [x] 定义动作空间 (MultiDiscrete)
  - [x] 定义奖励函数 (每个step -0.1, 完成任务 +10)
- [x] **算法实现 (Algorithm Implementation)**：
  - [x] 编写 `QLearningAgent` (src/agents/q_learning_agent.py)
  - [x] 编写训练循环 `main.py`
- [x] **实验与调优**：
  - [x] 优化状态空间 (State Space Simplification) -> 显著加速收敛
  - [x] 添加奖励塑造 (Reward Shaping) -> 引导机器人寻路
  - [ ] 采集数据与绘图 (Run final complete experiments)
- [ ] **论文撰写**

## 3. 技术选型与决策记录
| 决策项 | 选项 | 推荐/决定 | 原因 |
| :--- | :--- | :--- | :--- |
| **状态空间设计** | 全局坐标 / 相对方向 | **相对方向 (9 states)** | 极大地缩小了状态空间 (3600 -> 9)，使 Q-Learning 在几分钟内即可收敛，适合毕设快速演示。 |
| **编程语言** | Python / C++ / MATLAB | **Python** | AI领域绝对主流，库丰富，开发快。 |
| **仿真软件** | Python原生(Gymnasium) / Gazebo / Webots | **Python原生** | 本课题侧重“调度逻辑”而非“物理控制”。Python原生环境训练速度快，易于实现大规模步数训练。 |
| **机器学习框架** | PyTorch / TensorFlow | **PyTorch** | 社区活跃，调试方便，代码直观。 |

## 4. 下一步行动计划 (Next Steps)
1.  **用户必须手动安装 Python**：
    - 自动安装失败。请访问 [Python官网](https://www.python.org/downloads/) 下载。
    - **关键**：安装向导第一页务必勾选 **"Add Python to PATH"**。
2.  **验证安装**：重启 VS Code，在终端输入 `python --version`。
3.  **安装依赖**：运行 `pip install -r requirements.txt`。
4.  **运行程序**：运行 `python main.py`。
