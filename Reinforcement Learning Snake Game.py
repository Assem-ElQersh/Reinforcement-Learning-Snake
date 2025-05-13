import pickle
import random
import tkinter as tk
from tkinter import ttk
import numpy as np

# Game settings
GAME_WIDTH = 700
GAME_HEIGHT = 600
SNAKE_SIZE = 20
FOOD_SIZE = 20
BACKGROUND_COLOR = "#000000"
ACTIONS = ['Left', 'Right', 'Up', 'Down']

# Parameters for Q-learning
GAMMA = 0.9  # Discount factor
ALPHA = 0.1  # Learning rate
# EPSILON = 0.1  # Exploration rate
EPSILON = 0.5  # Initial exploration rate
EPSILON_DECAY = 0.99  # Exploration decay rate
EPSILON_MIN = 0.01  # Minimum exploration rate

# Game speeds (in milliseconds)
SPEEDS = {
    "Slow": 200,
    "Normal": 100,
    "Fast": 50
}

# Reward values
REWARDS = {
    "FOOD": 20,
    "DEATH": -100,
    "STEP": -0.1,
    "CLOSER_TO_FOOD": 0.5,
    "FARTHER_FROM_FOOD": -0.5
}

class QLearningSnakeAgent:
    def __init__(self, state_size, action_size, q_table_file="q_table.pkl"):
        self.q_table = {}  # Initialize Q-table
        self.state_size = state_size
        self.action_size = action_size
        self.q_table_file = q_table_file  # File where Q-table will be saved/loaded from

        # Try to load the Q-table if it exists
        self.load_q_table()

    def save_q_table(self):
        """Save the Q-table to a file."""
        with open(self.q_table_file, 'wb') as file:
            pickle.dump(self.q_table, file)
        print("Q-table saved.")

    def load_q_table(self):
        """Load the Q-table from a file if it exists."""
        try:
            with open(self.q_table_file, 'rb') as file:
                self.q_table = pickle.load(file)
            print("Q-table loaded from file.")
        except FileNotFoundError:
            print("No Q-table found, starting with a new table.")

    def get_state(self, snake, food):
        """Convert game state (snake and food positions) into a simplified representation."""
        head_x, head_y = snake[0]
        food_x, food_y = food

        # Determine dangers
        dangers = {
            'left': (head_x - SNAKE_SIZE, head_y) in snake or head_x - SNAKE_SIZE < 0,
            'right': (head_x + SNAKE_SIZE, head_y) in snake or head_x + SNAKE_SIZE >= GAME_WIDTH,
            'up': (head_x, head_y - SNAKE_SIZE) in snake or head_y - SNAKE_SIZE < 0,
            'down': (head_x, head_y + SNAKE_SIZE) in snake or head_y + SNAKE_SIZE >= GAME_HEIGHT,
        }

        # Calculate relative food position
        food_direction = (
            head_x < food_x,  # Food is to the right
            head_x > food_x,  # Food is to the left
            head_y < food_y,  # Food is below
            head_y > food_y,  # Food is above
        )

        # Calculate Manhattan distance to food
        manhattan_distance = abs(head_x - food_x) + abs(head_y - food_y)

        # Return state tuple
        state = (
            dangers['left'],
            dangers['right'],
            dangers['up'],
            dangers['down'],
            *food_direction,
            manhattan_distance < 100,  # Food is close
            manhattan_distance < 200,  # Food is medium distance
            manhattan_distance < 300,  # Food is far
        )
        return state

    def get_valid_actions(self, state):
        """Get list of valid actions that won't cause immediate collision."""
        valid_actions = []
        dangers = {
            'Left': state[0],
            'Right': state[1],
            'Up': state[2],
            'Down': state[3]
        }
        
        for action in ACTIONS:
            if not dangers[action]:  # If there's no danger in that direction
                valid_actions.append(action)
                
        return valid_actions if valid_actions else ACTIONS  # If no valid moves, return all actions

    def choose_action(self, state):
        """Choose an action based on epsilon-greedy policy, avoiding invalid moves."""
        valid_actions = self.get_valid_actions(state)
        
        if np.random.uniform(0, 1) < EPSILON:
            return random.choice(valid_actions)  # Exploration among valid actions
        else:
            q_values = self.q_table.get(state, [0] * len(ACTIONS))
            # Filter Q-values to only consider valid actions
            valid_q_values = [(i, q) for i, q in enumerate(q_values) if ACTIONS[i] in valid_actions]
            if valid_q_values:
                # Choose the best valid action
                best_action_idx = max(valid_q_values, key=lambda x: x[1])[0]
                return ACTIONS[best_action_idx]
            else:
                # If no valid actions in Q-table, choose randomly from valid actions
                return random.choice(valid_actions)

    def learn(self, state, action, reward, next_state):
        """Update Q-table using Q-learning rule."""
        action_index = ACTIONS.index(action)
        q_values = self.q_table.get(state, [0] * len(ACTIONS))

        # Get the Q-value for the current state-action pair
        current_q = q_values[action_index]
        next_q_values = self.q_table.get(next_state, [0] * len(ACTIONS))

        # Q-learning update rule
        q_values[action_index] = current_q + ALPHA * (reward + GAMMA * np.max(next_q_values) - current_q)

        # Store updated Q-values in the table
        self.q_table[state] = q_values

class SnakeGame:
    def __init__(self, root, agent):
        self.root = root
        self.root.title("Snake Game - Reinforcement Learning")
        self.score = 0
        self.high_score = 0
        self.game_over = False
        self.agent = agent
        self.food_position = None
        self.current_speed = SPEEDS["Normal"]
        self.training_mode = True

        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create control panel
        self.control_panel = ttk.Frame(self.main_frame)
        self.control_panel.pack(fill=tk.X, padx=5, pady=5)

        # Score display
        self.score_label = ttk.Label(self.control_panel, text="Score: 0")
        self.score_label.pack(side=tk.LEFT, padx=5)

        # High score display
        self.high_score_label = ttk.Label(self.control_panel, text="High Score: 0")
        self.high_score_label.pack(side=tk.LEFT, padx=5)

        # Speed control
        self.speed_var = tk.StringVar(value="Normal")
        self.speed_label = ttk.Label(self.control_panel, text="Speed:")
        self.speed_label.pack(side=tk.LEFT, padx=5)
        self.speed_combo = ttk.Combobox(self.control_panel, textvariable=self.speed_var, 
                                      values=list(SPEEDS.keys()), state="readonly", width=10)
        self.speed_combo.pack(side=tk.LEFT, padx=5)
        self.speed_combo.bind("<<ComboboxSelected>>", self.change_speed)

        # Training mode toggle
        self.training_var = tk.BooleanVar(value=True)
        self.training_check = ttk.Checkbutton(self.control_panel, text="Training Mode", 
                                            variable=self.training_var)
        self.training_check.pack(side=tk.LEFT, padx=5)

        # Restart button
        self.restart_button = ttk.Button(self.control_panel, text="Restart", command=self.restart_game)
        self.restart_button.pack(side=tk.RIGHT, padx=5)

        # Set up the game canvas
        self.canvas = tk.Canvas(self.main_frame, bg=BACKGROUND_COLOR, height=GAME_HEIGHT, width=GAME_WIDTH)
        self.canvas.pack(padx=5, pady=5)

        # Initialize snake and food
        self.snake = [(100, 100), (80, 100), (60, 100)]
        self.food = None
        self.direction = "Right"
        self.create_objects()

        # Bind keyboard controls
        self.root.bind("<Left>", lambda e: self.change_direction("Left"))
        self.root.bind("<Right>", lambda e: self.change_direction("Right"))
        self.root.bind("<Up>", lambda e: self.change_direction("Up"))
        self.root.bind("<Down>", lambda e: self.change_direction("Down"))
        self.root.bind("<space>", lambda e: self.restart_game())

        # Start the game
        self.run_game()

    def change_speed(self, event=None):
        """Change the game speed."""
        self.current_speed = SPEEDS[self.speed_var.get()]

    def update_score_display(self):
        """Update the score display."""
        self.score_label.config(text=f"Score: {self.score}")
        if self.score > self.high_score:
            self.high_score = self.score
            self.high_score_label.config(text=f"High Score: {self.high_score}")

    def create_objects(self):
        """Create snake and food objects in the canvas."""
        self.snake_body = []
        for segment in self.snake:
            self.snake_body.append(self.canvas.create_rectangle(segment[0], segment[1], segment[0] + SNAKE_SIZE,
                                                                segment[1] + SNAKE_SIZE, fill="green"))

        self.spawn_food()

    def spawn_food(self):
        """Place the food at a random position on the canvas."""
        food_x = random.randint(0, (GAME_WIDTH // FOOD_SIZE) - 1) * FOOD_SIZE
        food_y = random.randint(0, (GAME_HEIGHT // FOOD_SIZE) - 1) * FOOD_SIZE
        self.food = self.canvas.create_rectangle(food_x, food_y, food_x + FOOD_SIZE, food_y + FOOD_SIZE, fill="red")
        self.food_position = (food_x, food_y)

    def move_snake(self):
        """Move the snake by updating its coordinates."""
        x, y = self.snake[0]

        if self.direction == "Left":
            x -= SNAKE_SIZE
        elif self.direction == "Right":
            x += SNAKE_SIZE
        elif self.direction == "Up":
            y -= SNAKE_SIZE
        elif self.direction == "Down":
            y += SNAKE_SIZE

        new_head = (x, y)
        self.snake = [new_head] + self.snake

        # Always remove the last part (we'll handle growth separately)
        self.snake.pop()

        # Update snake body on canvas
        self.update_snake_body()

    def update_snake_body(self):
        """Update the canvas representation of the snake."""
        # Clear the canvas and redraw the snake
        for i, segment in enumerate(self.snake):
            if i < len(self.snake_body):
                self.canvas.coords(self.snake_body[i], segment[0], segment[1],
                                   segment[0] + SNAKE_SIZE, segment[1] + SNAKE_SIZE)
            else:
                self.snake_body.append(self.canvas.create_rectangle(segment[0], segment[1], 
                                                                    segment[0] + SNAKE_SIZE,
                                                                    segment[1] + SNAKE_SIZE, fill="green"))

    def food_collision(self):
        """Check if the snake's head has collided with the food."""
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food_position

        if head_x == food_x and head_y == food_y:
            self.canvas.delete(self.food)
            self.spawn_food()
            return True  # Collision detected
        return False

    def check_collisions(self):
        """Check if the snake has collided with the wall or itself."""
        head_x, head_y = self.snake[0]

        # Check wall collisions
        if head_x < 0 or head_x >= GAME_WIDTH or head_y < 0 or head_y >= GAME_HEIGHT:
            self.game_over = True

        # Check self-collision
        if (head_x, head_y) in self.snake[1:]:
            self.game_over = True

    def run_game(self):
        """Main game loop with Reinforcement Learning."""
        if not self.game_over:
            state = self.agent.get_state(self.snake, self.food_position)
            
            # Only use RL agent in training mode
            if self.training_var.get():
                action = self.agent.choose_action(state)
                self.direction = action
            else:
                # Manual control in non-training mode
                pass

            # Move the snake
            self.move_snake()

            # Check for collisions
            self.check_collisions()

            # Check for food collision
            ate_food = self.food_collision()

            # If food is eaten, grow the snake
            if ate_food:
                self.snake.append(self.snake[-1])
                self.score += 10
                self.update_score_display()

            # Update Q-table if in training mode
            if self.training_var.get():
                next_state = self.agent.get_state(self.snake, self.food_position)
                reward = self.calculate_reward(ate_food)
                self.agent.learn(state, action, reward, next_state)

            self.root.after(self.current_speed, self.run_game)
        else:
            self.end_game()

    def calculate_reward(self, ate_food):
        """Enhanced reward system for RL."""
        if self.game_over:
            return REWARDS["DEATH"]
        
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food_position
        
        # Calculate Manhattan distance to food
        current_distance = abs(head_x - food_x) + abs(head_y - food_y)
        
        # Calculate previous distance
        prev_head_x, prev_head_y = self.snake[1]
        prev_distance = abs(prev_head_x - food_x) + abs(prev_head_y - food_y)
        
        # Base reward
        reward = REWARDS["STEP"]
        
        # Reward for getting closer to food
        if current_distance < prev_distance:
            reward += REWARDS["CLOSER_TO_FOOD"]
        elif current_distance > prev_distance:
            reward += REWARDS["FARTHER_FROM_FOOD"]
            
        # Reward for eating food
        if ate_food:
            reward += REWARDS["FOOD"]
            
        return reward

    def end_game(self):
        """Handle game over event."""
        self.canvas.create_text(GAME_WIDTH / 2, GAME_HEIGHT / 2,
                              text=f"Game Over!\nScore: {self.score}\nHigh Score: {self.high_score}",
                              fill="white", font=("Arial", 24), justify=tk.CENTER)
        
        # Save Q-table when the game ends
        if self.training_var.get():
            self.agent.save_q_table()

        global EPSILON
        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY
        # Restart the game for continuous learning
        self.root.after(2000, self.restart_game)

    def restart_game(self):
        """Restart the game with initial values."""
        self.score = 0
        self.game_over = False
        self.update_score_display()
        
        # Clear canvas
        self.canvas.delete("all")
        
        # Reset snake and food
        self.snake = [(100, 100), (80, 100), (60, 100)]
        self.direction = "Right"
        self.create_objects()
        
        # Start the game again
        self.run_game()

    def change_direction(self, new_direction):
        """Change the snake's direction if valid."""
        if not self.training_var.get():  # Only allow manual control in non-training mode
            opposites = {"Left": "Right", "Right": "Left", "Up": "Down", "Down": "Up"}
            if new_direction != opposites.get(self.direction):
                self.direction = new_direction

if __name__ == "__main__":
    root = tk.Tk()
    
    # Create and configure the agent
    state_size = 11  # Updated state size with new features
    action_size = len(ACTIONS)
    agent = QLearningSnakeAgent(state_size, action_size)
    
    # Create and start the game
    game = SnakeGame(root, agent)
    
    # Start the main loop
    root.mainloop()


