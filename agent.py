import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

WIDTH, HEIGHT = 400, 400  
BLOCK_SIZE = 20            
FPS = 10                   

BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class SnakeGame:
    def __init__(self):
        pygame.init()
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Snake Game with DQN')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.snake = [[60, 60], [40, 60], [20, 60]]  
        self.direction = [BLOCK_SIZE, 0] 
        self.food = self.generate_food()
        self.done = False
        return self.get_state()

    def generate_food(self):
        while True:
            food = [
                random.randrange(0, WIDTH // BLOCK_SIZE) * BLOCK_SIZE,
                random.randrange(0, HEIGHT // BLOCK_SIZE) * BLOCK_SIZE,
            ]
            if food not in self.snake:
                return food

    def get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        state = [
            head_x < food_x, 
            head_x > food_x, 
            head_y < food_y, 
            head_y > food_y,  
            self.direction == [-BLOCK_SIZE, 0], 
            self.direction == [BLOCK_SIZE, 0],   
            self.direction == [0, -BLOCK_SIZE], 
            self.direction == [0, BLOCK_SIZE],   
        ]
        return np.array(state, dtype=np.float32)

    def step(self, action):
        opposite_directions = {0: 1, 1: 0, 2: 3, 3: 2}
        if action != opposite_directions.get(self.get_direction_index(), action):
            self.direction = [
                [-BLOCK_SIZE, 0],  
                [BLOCK_SIZE, 0],   
                [0, -BLOCK_SIZE],  
                [0, BLOCK_SIZE],   
            ][action]

        new_head = [
            self.snake[0][0] + self.direction[0],
            self.snake[0][1] + self.direction[1]
        ]
        self.snake.insert(0, new_head)  

        if new_head == self.food:
            self.food = self.generate_food()  
            reward = 10  
        else:
            self.snake.pop() 
            reward = -0.1  

        if self.is_collision(new_head):
            self.done = True
            reward = -10  

        return self.get_state(), reward, self.done

    def is_collision(self, head):
        return (
            head[0] < 0 or head[0] >= WIDTH or
            head[1] < 0 or head[1] >= HEIGHT or
            head in self.snake[1:]
        )

    def get_direction_index(self):
        directions = [
            [-BLOCK_SIZE, 0],  
            [BLOCK_SIZE, 0],   
            [0, -BLOCK_SIZE],  
            [0, BLOCK_SIZE],   
        ]
        return directions.index(self.direction)

    def render(self):
        self.display.fill(BLACK) 
        for segment in self.snake:
            pygame.draw.rect(self.display, GREEN, (*segment, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, RED, (*self.food, BLOCK_SIZE, BLOCK_SIZE))
        pygame.display.update()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward + (1 - done) * self.gamma * torch.max(self.target_model(torch.FloatTensor(next_state).unsqueeze(0)))
            target_f = self.model(torch.FloatTensor(state).unsqueeze(0)).detach()
            target_f[0][action] = target

            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(torch.FloatTensor(state).unsqueeze(0)), target_f)
            loss.backward()
            self.optimizer.step()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train():
    game = SnakeGame()
    agent = DQNAgent(state_size=8, action_size=4)
    episodes = 1000

    plt.ion()
    fig, ax = plt.subplots()
    scores = []

    for episode in range(episodes):
        state = game.reset()
        total_reward = 0

        while True:
            game.handle_events()
            action = agent.act(state)
            next_state, reward, done = game.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                agent.update_target_model()
                scores.append(len(game.snake) - 3)
                ax.clear()
                ax.plot(scores, label='Score per Episode')
                ax.legend()
                plt.pause(0.01)  
                print(f"Episode: {episode + 1}, Score: {len(game.snake) - 3}, Total Reward: {total_reward}")
                break

            agent.replay()
            game.render()
            game.clock.tick(FPS)

        agent.decay_epsilon()

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    train()
