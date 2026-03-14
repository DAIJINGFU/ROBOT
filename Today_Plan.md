# 毕业设计今日冲刺计划 (Today's Sprint Plan)

**目标**：完成代码调试，跑通多智能体协同调度实验，获取初步数据图表，满足任务书关于“吞吐量-等待时间优化”的基本演示要求。

## 0. 关键问题修复 (Immediate Fixes) - 预计耗时: 30分钟
**现状分析**：
当前 `main.py` 中的状态定义 (`make_state_hashable`) 仅包含机器人自身位置和任务完成状态，**缺少任务位置信息**。
- **后果**：由于任务位置在每次 `reset()` 时随机生成，智能体无法通过 Q-learning 记住任务在哪里，导致无法收敛（学不到东西）。
- **解决方案**：为了展示效果，暂时将**任务位置固定**，或者将最近任务的相对坐标加入状态。建议先采用**固定任务位置**，确保模型能快速收敛。

## 1. 上午：核心代码完善 (Morning: Core Logic)
- [ ] **增强环境 (`robot_env.py`)**：
    - 增加 `fixed_tasks` 标志，支持训练时固定任务位置。
    - 增加指标统计：在 `info` 中返回当前回合的“累积等待时间”和“完成任务数”。
- [ ] **完善主循环 (`main.py`)**：
    - 增加数据记录功能 (Python list / CSV)。
    - 记录指标：`Episode`, `Steps`, `Total Reward`, `Avg Waiting Time`, `Throughput` (任务数/Steps).

## 2. 下午：实验与数据采集 (Afternoon: Experiments)
- [ ] **实验 A (Baseline)**：
    - 设置：奖励函数 $R = \text{Completion}(+10) - \text{TimeCost}(0.1)$。
    - 训练 10000 steps (或 500 episodes)，观察收敛曲线。
- [ ] **实验 B (多目标对比)**：
    - 调整奖励权重，例如加大时间惩罚 $R = \text{Completion}(+10) - \text{TimeCost}(0.5)$，观察机器人是否更倾向于快速完成任务（牺牲部分协同或更激进）。
    - *注：无模型 RL 调节参数其实就是调节 Reward。*

## 3. 晚上：绘图与整理 (Evening: Visualization)
- [ ] **编写绘图脚本 (`plot_results.py`)**：
    - 绘制 Reward 变化曲线 (验证收敛性)。
    - 绘制 Throughput vs. Waiting Time 散点图 (展示多目标优化效果)。
- [ ] **整理文档**：
    - 将实验截图和分析写入论文草稿。

---
**我已准备好为你修复 `robot_env.py` 和 `main.py` 以便立即开始训练。是否执行？**
