# main.py
import multiprocessing
import os
import sys

def run_model(module_name):
    # Import the module and run its train() function.
    module = __import__(module_name)
    module.train()

if __name__ == "__main__":
    print("Select models to train:")
    print("1: DQN Model")
    print("2: Random Model")
    choices = input("Enter model numbers separated by comma (e.g., 1,2): ")
    selected = [choice.strip() for choice in choices.split(",")]
    processes = []
    for choice in selected:
        if choice == "1":
            p = multiprocessing.Process(target=run_model, args=("model_dqn",))
            processes.append(p)
        elif choice == "2":
            p = multiprocessing.Process(target=run_model, args=("model_random",))
            processes.append(p)
    print(f"Starting {len(processes)} model(s)...")
    for p in processes:
        p.start()
    for p in processes:
        p.join()
