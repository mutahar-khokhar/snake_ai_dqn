# Deep Q-Learning Snake Game

This repository implements a **Deep Q-Learning** (DQN) agent to play the **Snake Game** using **Reinforcement Learning**.

---

## Files:
- **`snake.py`**: The Snake Game logic built with **Pygame**.
- **`model_dqn.py`**: The **DQN** model using **PyTorch** for decision-making.
- **`model_random.py`**: A **Random Agent** as a baseline.
- **`main.py`**: Runs the game and trains the DQN agent.
- **`game_settings.py`**: Contains game settings along with reward and penalty setting.

---

## Features:
- **Deep Q-Learning** for decision-making.
- **Epsilon-Greedy Exploration** for balanced learning.
- **Reward Shaping** to encourage chasing fruit and avoiding collisions.
- **Real-time training feedback** logged during episodes.
