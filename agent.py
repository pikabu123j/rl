# agent.py

import torch
import random
import numpy as np
from collections import deque
from model import TensorQNetwork
from config import CONFIG

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.gamma = CONFIG['gamma']
        self.epsilon = CONFIG['epsilon_start']
        self.epsilon_min = CONFIG['epsilon_end']
        self.epsilon_decay = CONFIG['epsilon_decay']
        self.batch_size = CONFIG['batch_size']

        self.memory = deque(maxlen=CONFIG['memory_size'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = TensorQNetwork(state_dim, action_dim, CONFIG['hidden_dim']).to(self.device)
        self.target_network = TensorQNetwork(state_dim, action_dim, CONFIG['hidden_dim']).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=CONFIG['learning_rate'])

        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, CONFIG['action_dim'] - 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_network(state).argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        current_q = self.q_network(states).gather(1, actions)
        next_q = self.target_network(next_states).max(1)[0].unsqueeze(1).detach()
        expected_q = rewards + (1 - dones) * self.gamma * next_q

        loss = torch.nn.functional.mse_loss(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
