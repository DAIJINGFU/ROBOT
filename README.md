# Robot Multi-Agent Task Allocation System

This project implements a multi-agent reinforcement learning system for coordinate task allocation in a grid environment.

## Features
- **Algorithm**: Independent Q-Learning with Static Partitioning.
- **Environment**: Custom 10x10 Grid World with dynamic task generation.
- **Optimization**: Minimized state space for rapid convergence (~2000 episodes).
- **Results**: Achieved near-optimal path planning (17 steps/episode) and high throughput.

## Usage

### Prerequisites
- Python 3.11+
- Gymnasium
- NumPy
- Matplotlib

### Installation
```bash
pip install -r requirements.txt
```

### Running the Experiment
Execute the main script:
```bash
python main_final.py
```
This will train the agents and generate performance plots (`paper_results.png`).

## Repository Structure
- `src/`: Source code for agents and environments.
- `main_final.py`: Main entry point for the optimized system.
- `paper_results.png`: Generated performance metrics.
