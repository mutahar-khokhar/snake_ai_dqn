# model_random.py
import pygame
import random
import csv
import os
from snake import SnakeGame

MODEL_NAME = "Random"

def log_to_csv(test_number, model_name, episode, score, record, filename="model_random.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as csv_file:
        fieldnames = ["test_number", "model_name", "episode", "score", "record"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "test_number": test_number,
            "model_name": model_name,
            "episode": episode,
            "score": score,
            "record": record
        })

def train():
    pygame.init()
    game = SnakeGame(display=True, window_title="Random Model")
    record = 0
    n_episodes = 1000  # Adjust as needed
    test_number = 1
    directions = ["UP", "DOWN", "LEFT", "RIGHT"]
    for episode in range(1, n_episodes + 1):
        game.reset()
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            new_dir = random.choice(directions)
            _, _, done = game.step(new_dir)
            game.render()
            if done:
                break
        if game.score > record:
            record = game.score
        print(f"Random Model Episode {episode} Score: {game.score} Record: {record}")
        log_to_csv(test_number, MODEL_NAME, episode, game.score, record, filename="model_random.csv")
    pygame.quit()

if __name__ == "__main__":
    train()
