# model_dqn.py
import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import csv
from collections import deque, namedtuple
from snake import SnakeGame, SCREEN_WIDTH, SCREEN_HEIGHT, BLOCK_SIZE

MODEL_NAME = "DQN"

# --- Extended Agent State Extraction ---
def get_agent_state(game):
    """
    Returns a 14-dimensional state as a numpy array:
      Indices 0-2:  [danger_straight, danger_right, danger_left]
      Indices 3-6:  [dir_up, dir_down, dir_left, dir_right]
      Indices 7-10: [target fruit relative position: left, right, up, down]
      Indices 11-13: [one-hot fruit type: apple, banana, cherry]
      
    The target fruit is chosen greedily based on (fruit.value / (distance+ε)).
    """
    head = game.snake.get_head()

    # Select the target fruit using a greedy ratio.
    best_fruit = None
    best_ratio = -1
    for fruit in game.fruits:
        dist = np.linalg.norm(np.array(head) - np.array(fruit.position))
        ratio = fruit.value / (dist + 1e-5)  # avoid division by zero
        if ratio > best_ratio:
            best_ratio = ratio
            best_fruit = fruit

    # Relative position flags for the target fruit:
    fruit_left  = best_fruit.position[0] < head[0]
    fruit_right = best_fruit.position[0] > head[0]
    fruit_up    = best_fruit.position[1] < head[1]
    fruit_down  = best_fruit.position[1] > head[1]

    # One-hot encoding for fruit type: apple, banana, cherry.
    if best_fruit.type == "apple":
        fruit_type = [1, 0, 0]
    elif best_fruit.type == "banana":
        fruit_type = [0, 1, 0]
    else:
        fruit_type = [0, 0, 1]

    # Get danger flags – using a helper _check_collision (monkey-patched below)
    head_point = head
    point_l = (head[0] - BLOCK_SIZE, head[1])
    point_r = (head[0] + BLOCK_SIZE, head[1])
    point_u = (head[0], head[1] - BLOCK_SIZE)
    point_d = (head[0], head[1] + BLOCK_SIZE)

    # Current direction flags:
    dir_l = game.snake.direction == "LEFT"
    dir_r = game.snake.direction == "RIGHT"
    dir_u = game.snake.direction == "UP"
    dir_d = game.snake.direction == "DOWN"

    # Danger in directions depends on current movement.
    if game.snake.direction == "RIGHT":
        danger_straight = game._check_collision(point_r)
        danger_right    = game._check_collision(point_d)
        danger_left     = game._check_collision(point_u)
    elif game.snake.direction == "LEFT":
        danger_straight = game._check_collision(point_l)
        danger_right    = game._check_collision(point_u)
        danger_left     = game._check_collision(point_d)
    elif game.snake.direction == "UP":
        danger_straight = game._check_collision(point_u)
        danger_right    = game._check_collision(point_r)
        danger_left     = game._check_collision(point_l)
    else:  # DOWN
        danger_straight = game._check_collision(point_d)
        danger_right    = game._check_collision(point_l)
        danger_left     = game._check_collision(point_r)

    state = [
        int(danger_straight),
        int(danger_right),
        int(danger_left),
        int(dir_u),
        int(dir_d),
        int(dir_l),
        int(dir_r),
        int(fruit_left),
        int(fruit_right),
        int(fruit_up),
        int(fruit_down)
    ]
    # Append the one-hot fruit type (3 elements) to get a 14-dim state.
    state.extend(fruit_type)
    return np.array(state, dtype=int), best_fruit

# --- Monkey-Patch SnakeGame to add a collision-checker ---
def _check_collision(self, pt):
    # Check boundaries.
    if pt[0] < 0 or pt[0] >= SCREEN_WIDTH or pt[1] < 0 or pt[1] >= SCREEN_HEIGHT:
        return True
    # Check self-collision.
    if pt in self.snake.segments:
        return True
    # Check obstacles.
    for obs in self.obstacles:
        if pt == obs.position:
            return True
    return False

SnakeGame._check_collision = _check_collision

# --- Q-Network and Agent ---
class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
    def save(self, file_name='model_dqn.pth'):
        torch.save(self.state_dict(), file_name)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0      # randomness factor
        self.gamma = 0.9      # discount rate
        self.memory = ReplayMemory(100_000)
        self.batch_size = 1000
        self.learning_rate = 0.001
        # Updated input size: 14 instead of 11.
        self.model = QNet(14, 128, 3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
    
    def get_action(self, state):
        # Decaying epsilon strategy.
        self.epsilon = max(80 - self.n_games, 0)
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move
    
    def train_short_memory(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float)
        done = torch.tensor([done], dtype=torch.bool)
        
        pred = self.model(state)
        target = pred.clone().detach()
        action_idx = torch.argmax(action).item()
        if done:
            Q_new = reward
        else:
            Q_new = reward + self.gamma * torch.max(self.model(next_state))
        target[action_idx] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()
    
    def train_long_memory(self):
        if len(self.memory) < self.batch_size:
            mini_sample = list(self.memory.memory)
        else:
            mini_sample = self.memory.sample(self.batch_size)
        
        if len(mini_sample) == 0:
            return
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        states = torch.tensor(np.array(states), dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float)
        dones = torch.tensor(np.array(dones), dtype=torch.bool)
        
        pred = self.model(states)
        target = pred.clone().detach()
        for i in range(len(mini_sample)):
            action_idx = torch.argmax(actions[i]).item()
            if dones[i]:
                Q_new = rewards[i]
            else:
                Q_new = rewards[i] + self.gamma * torch.max(self.model(next_states[i]))
            target[i][action_idx] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()

# --- Extended CSV Logging ---
def log_to_csv(test_number, model_name, episode, score, record, epsilon, filename="model_dqn.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as csv_file:
        fieldnames = ["test_number", "model_name", "episode", "score", "record", "epsilon"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "test_number": test_number,
            "model_name": model_name,
            "episode": episode,
            "score": score,
            "record": record,
            "epsilon": epsilon
        })

# --- Training Loop ---
def train():
    pygame.init()
    agent = Agent()
    game = SnakeGame(display=True, window_title="DQN Model")
    record = 0
    n_episodes = 1000  # Adjust as needed
    test_number = 1  # For example, if you run multiple tests
    for episode in range(1, n_episodes + 1):
        game.reset()
        state, target_fruit = get_agent_state(game)
        # For reward shaping: compute initial distance from head to target fruit.
        head = np.array(game.snake.get_head())
        target = np.array(target_fruit.position)
        old_distance = np.linalg.norm(head - target)
        
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            action = agent.get_action(state)
            
            # Map action (one-hot: [straight, right, left]) to new direction.
            directions = ["RIGHT", "DOWN", "LEFT", "UP"]
            idx = directions.index(game.snake.direction)
            if action == [1, 0, 0]:
                new_dir = directions[idx]
            elif action == [0, 1, 0]:
                new_dir = directions[(idx + 1) % 4]
            elif action == [0, 0, 1]:
                new_dir = directions[(idx - 1) % 4]
            else:
                new_dir = directions[idx]
            
            state_old = state
            _, reward, done = game.step(new_dir)
            # Get new state and target fruit.
            state, target_fruit = get_agent_state(game)
            new_head = np.array(game.snake.get_head())
            target = np.array(target_fruit.position)
            new_distance = np.linalg.norm(new_head - target)
            
            # --- Greedy Reward Shaping ---
            # Bonus for moving closer to the target fruit.
            delta = old_distance - new_distance
            reward += 0.1 * delta  # adjust multiplier as needed
            old_distance = new_distance
            
            # --- Extra Penalty for "Chasing Itself" ---
            # Compute local density: count how many segments (excluding head) lie within 2*BLOCK_SIZE.
            new_head_np = np.array(new_head)
            density = sum(
                1 for seg in game.snake.segments[1:]
                if np.linalg.norm(new_head_np - np.array(seg)) < (2 * BLOCK_SIZE)
            )
            # If density is high, penalize (e.g. -0.5 per nearby segment beyond 1).
            if density > 1:
                reward -= 0.5 * (density - 1) ** 2
            
            agent.train_short_memory(state_old, action, reward, state, done)
            agent.memory.push(state_old, action, reward, state, done)
            game.render()
            if done:
                break
        
        agent.n_games += 1
        agent.train_long_memory()
        if game.score > record:
            record = game.score
            agent.model.save()
        print(f"DQN Episode {episode} Score: {game.score} Record: {record} Epsilon: {agent.epsilon}")
        log_to_csv(test_number, MODEL_NAME, episode, game.score, record, agent.epsilon, filename="model_dqn.csv")
    pygame.quit()

if __name__ == "__main__":
    train()
