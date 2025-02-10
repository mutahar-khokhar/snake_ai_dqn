# snake.py
import pygame
import random
import sys
from game_settings import (
    BASE_SPEED, SPEED_INCREMENT,
    FRUIT_SCORE_MULTIPLIER, STEP_PENALTY, COLLISION_PENALTY,
    FRUIT_COUNT,
    OBSTACLE_SCORE_THRESHOLD, OBSTACLE_INCREMENT, INITIAL_OBSTACLE_COUNT
)

pygame.init()

# === Global Display Settings ===
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
BLOCK_SIZE = 20

# Colors
GREEN      = (0, 255, 0)
DARK_GREEN = (0, 200, 0)
RED        = (255, 0, 0)
YELLOW     = (255, 255, 0)
PINK       = (255, 105, 180)
WHITE      = (255, 255, 255)
BLACK      = (0, 0, 0)
GRAY       = (100, 100, 100)

# Fonts
font_small = pygame.font.SysFont("comicsansms", 18)
font_large = pygame.font.SysFont("comicsansms", 36)

# Fruit types: Each fruit has a color and a score value.
fruit_types = {
    "apple": {"color": RED, "score": 1},
    "banana": {"color": YELLOW, "score": 2},
    "cherry": {"color": PINK, "score": 3}
}

def get_random_position():
    x = random.randrange(0, SCREEN_WIDTH // BLOCK_SIZE) * BLOCK_SIZE
    y = random.randrange(0, SCREEN_HEIGHT // BLOCK_SIZE) * BLOCK_SIZE
    return (x, y)

# --- Game Object Classes ---
class Fruit:
    def __init__(self):
        # Randomly choose a fruit type.
        self.type = random.choice(list(fruit_types.keys()))
        self.color = fruit_types[self.type]["color"]
        self.value = fruit_types[self.type]["score"]
        self.position = get_random_position()
    
    def draw(self, surface):
        center = (self.position[0] + BLOCK_SIZE // 2, self.position[1] + BLOCK_SIZE // 2)
        radius = BLOCK_SIZE // 2 - 2
        pygame.draw.circle(surface, self.color, center, radius)
        # Display the fruit's score value.
        text = font_small.render(str(self.value), True, BLACK)
        text_rect = text.get_rect(center=center)
        surface.blit(text, text_rect)

class Obstacle:
    def __init__(self):
        self.position = get_random_position()
    
    def draw(self, surface):
        pygame.draw.rect(surface, GRAY, (self.position[0], self.position[1], BLOCK_SIZE, BLOCK_SIZE))

class Snake:
    def __init__(self):
        # Start with three segments at the center.
        self.segments = [
            (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2),
            (SCREEN_WIDTH // 2 - BLOCK_SIZE, SCREEN_HEIGHT // 2),
            (SCREEN_WIDTH // 2 - 2 * BLOCK_SIZE, SCREEN_HEIGHT // 2)
        ]
        self.direction = "RIGHT"
    
    def move(self):
        head_x, head_y = self.segments[0]
        if self.direction == "RIGHT":
            head_x += BLOCK_SIZE
        elif self.direction == "LEFT":
            head_x -= BLOCK_SIZE
        elif self.direction == "UP":
            head_y -= BLOCK_SIZE
        elif self.direction == "DOWN":
            head_y += BLOCK_SIZE
        new_head = (head_x, head_y)
        self.segments.insert(0, new_head)
        self.segments.pop()  # Remove the tail (unless growing)
    
    def grow(self):
        # Append a new segment at the tail's location.
        self.segments.append(self.segments[-1])
    
    def draw(self, surface):
        for seg in self.segments:
            rect = pygame.Rect(seg[0], seg[1], BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(surface, GREEN, rect)
            pygame.draw.rect(surface, DARK_GREEN, rect, 1)
        # Display a label on the head.
        head_text = font_small.render("Developed by Mutahar", True, BLACK)
        head_rect = head_text.get_rect(center=(self.segments[0][0] + BLOCK_SIZE // 2,
                                                self.segments[0][1] + BLOCK_SIZE // 2))
        surface.blit(head_text, head_rect)
    
    def set_direction(self, new_direction):
        # Prevent reversing direction directly.
        opposite = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
        if new_direction == opposite.get(self.direction):
            return
        self.direction = new_direction
    
    def check_self_collision(self):
        head = self.segments[0]
        return head in self.segments[1:]
    
    def get_head(self):
        return self.segments[0]

# --- Game Environment Class ---
class SnakeGame:
    def __init__(self, display=True, window_title="Snake Game - Developed by Mutahar"):
        self.display = display
        if display:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption(window_title)
        self.clock = pygame.time.Clock()
        self.reset()
    
    def reset(self):
        self.snake = Snake()
        # Create the desired number of fruits.
        self.fruits = [Fruit() for _ in range(FRUIT_COUNT)]
        # Create the initial obstacles.
        self.obstacles = []
        for _ in range(INITIAL_OBSTACLE_COUNT):
            self.obstacles.append(Obstacle())
        self.score = 0
        self.obstacles_added = INITIAL_OBSTACLE_COUNT
        self.FPS = BASE_SPEED
        self.game_over_flag = False
        return self.get_state()
    
    def get_state(self):
        """
        Returns a dictionary representing the current state of the game.
        (This can be extended for AI purposes.)
        """
        return {
            "snake_head": self.snake.get_head(),
            "snake_segments": self.snake.segments,
            "fruits": [(fruit.position, fruit.value) for fruit in self.fruits],
            "obstacles": [obs.position for obs in self.obstacles],
            "score": self.score,
            "direction": self.snake.direction
        }
    
    def step(self, action=None):
        """
        Advances the game by one step.
          - action: (Optional) new direction (e.g., "UP", "DOWN", etc.)
        Returns a tuple: (state, reward, game_over_flag)
        """
        if action is not None:
            self.snake.set_direction(action)
        self.snake.move()
        head = self.snake.get_head()
        reward = 0

        # Check for collision with boundaries.
        if head[0] < 0 or head[0] >= SCREEN_WIDTH or head[1] < 0 or head[1] >= SCREEN_HEIGHT:
            self.game_over_flag = True
            reward = COLLISION_PENALTY
            return self.get_state(), reward, self.game_over_flag

        # Check self-collision.
        if self.snake.check_self_collision():
            self.game_over_flag = True
            reward = COLLISION_PENALTY
            return self.get_state(), reward, self.game_over_flag

        # Check collision with obstacles.
        # Instead of ending the game, we reduce the snake's length by 30%.
        collided_with_obstacle = False
        for obs in self.obstacles:
            if head == obs.position:
                collided_with_obstacle = True
                break
        if collided_with_obstacle:
            reward = COLLISION_PENALTY
            current_length = len(self.snake.segments)
            # Calculate new length (30% reduction; ensure at least 1 segment remains).
            new_length = max(1, int(current_length * 0.7))
            self.snake.segments = self.snake.segments[:new_length]
            # Note: We do not set game_over_flag here, so the game continues.
        
        # Check if the snake has eaten a fruit.
        fruit_eaten = None
        for fruit in self.fruits:
            if head == fruit.position:
                fruit_eaten = fruit
                self.score += fruit.value
                reward = fruit.value * FRUIT_SCORE_MULTIPLIER
                self.snake.grow()
                break
        if fruit_eaten:
            self.fruits.remove(fruit_eaten)
            new_fruit = Fruit()
            # Ensure the new fruit doesn't spawn on the snake or an obstacle.
            while new_fruit.position in self.snake.segments or any(new_fruit.position == obs.position for obs in self.obstacles):
                new_fruit.position = get_random_position()
            self.fruits.append(new_fruit)

        # Add obstacles more frequently.
        required_obstacles = INITIAL_OBSTACLE_COUNT + (self.score // OBSTACLE_SCORE_THRESHOLD) * OBSTACLE_INCREMENT
        while self.obstacles_added < required_obstacles:
            new_obs = Obstacle()
            # Ensure new obstacle doesn't spawn on the snake, a fruit, or an existing obstacle.
            while (new_obs.position in self.snake.segments or
                   any(new_obs.position == fruit.position for fruit in self.fruits) or
                   any(new_obs.position == obs.position for obs in self.obstacles)):
                new_obs.position = get_random_position()
            self.obstacles.append(new_obs)
            self.obstacles_added += 1

        # Increase game speed gradually.
        self.FPS = BASE_SPEED + (self.score // 10) * SPEED_INCREMENT

        reward -= STEP_PENALTY  # Apply a small step penalty.
        return self.get_state(), reward, self.game_over_flag

    def render(self):
        """Draws the current game state to the screen."""
        if not self.display:
            return
        self.screen.fill(BLACK)
        self.snake.draw(self.screen)
        for fruit in self.fruits:
            fruit.draw(self.screen)
        for obs in self.obstacles:
            obs.draw(self.screen)
        score_text = font_small.render("Score: " + str(self.score), True, WHITE)
        self.screen.blit(score_text, (10, 10))
        pygame.display.update()
        self.clock.tick(self.FPS)
    
    def close(self):
        if self.display:
            pygame.display.quit()

# --- For running the game directly ---
if __name__ == "__main__":
    game = SnakeGame(display=True)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # Optional key handling for manual control.
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    game.snake.set_direction("UP")
                elif event.key == pygame.K_DOWN:
                    game.snake.set_direction("DOWN")
                elif event.key == pygame.K_LEFT:
                    game.snake.set_direction("LEFT")
                elif event.key == pygame.K_RIGHT:
                    game.snake.set_direction("RIGHT")
        state, reward, game_over = game.step()  # No external action provided.
        game.render()
        if game_over:
            # For collisions with boundary or self, the game ends.
            pygame.time.delay(1000)
            # In this version, only collisions with boundaries or self cause game over.
            # (Obstacle collisions only reduce snake length.)
            game.reset()
