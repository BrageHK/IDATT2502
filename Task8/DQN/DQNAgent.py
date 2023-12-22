import numpy as np
import gymnasium as gym
from collections import deque

import torch
import torch.nn as nn
from QNetwork import QNetwork
from random import sample 


class QLearningAgent():
    def __init__(self, action_dim, observed_dim, learning_rate_initial, epsilon, gamma, env, hidden_dim, decay_rate = 0.001, batch=200, maxlen=2000):
        self.action_dim = action_dim
        self.observed_dim = observed_dim
        self.learning_rate_initial = learning_rate_initial
        self.learning_rate = learning_rate_initial
        self.epsilon = epsilon
        self.epsilon_initial = epsilon
        self.gamma = gamma
        self.batch = batch
        
        self.env = env
        self.this_episode = 0
        self.decay_rate = decay_rate
        
        self.Q_network = QNetwork(object_dim=observed_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.buffer = deque(maxlen=maxlen)
        self.optimizer = torch.optim.Adam(self.Q_network.parameters(), self.learning_rate)
        
        self.loss = nn.MSELoss()
    
    def decay_epsilon(self):
        self.epsilon = self.epsilon_initial * 1/(1 + self.decay_rate * self.this_episode)
        
    def act(self, value_func):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            return torch.argmax(value_func).item()
        
    def save(self, filename):
        torch.save(self.Q_network.state_dict(), filename)

    def load(self, filename):
        self.Q_network.load_state_dict(torch.load(filename))
        #self.Q_network.eval()
    
    def compute_loss(self, batch):
        # Unpack the batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.stack(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Compute current Q-values
        current_q_values = self.Q_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Compute next Q-values
        next_q_values = self.Q_network(next_states).max(1)[0]
        
        # Compute target Q-values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values.detach())

        return loss
  
    def train(self, episodes, render = False):
        if(render):
            self.epsilon = 0
            self.epsilon_initial = 0
            env = gym.make(self.env, render_mode="human")
        else:
            env = gym.make(self.env)
        for i in range(episodes):
            observation, info = env.reset()
            state = torch.tensor(observation)
            
            self.this_episode += 1

            self.decay_epsilon()
            
            terminated = False
            truncated = False
            score = 0
            while(True):
                value_func = self.Q_network.forward(state) # Predict
                action = self.act(value_func) # Get action
                
                observation, reward, terminated, truncated, _ = env.step(action) # Do action
                next_state  = torch.tensor(observation)
                score += reward
                
                done = 1 if truncated or terminated else 0
                self.buffer.append((state, action, reward, next_state, done))
                
                state = next_state
                
                if(done):
                    break
                if len(self.buffer) > self.batch:
                    random_sample = sample(self.buffer, self.batch)
                    
                    states = torch.stack([x[0] for x in random_sample])
                    actions = torch.tensor([x[1] for x in random_sample])
                    rewards = torch.tensor([x[2] for x in random_sample], dtype=torch.float32)
                    next_states = torch.stack([x[3] for x in random_sample])
                    dones = torch.tensor([x[4] for x in random_sample], dtype=torch.float32)
                    
                    target_max, _ = self.Q_network(next_states).max(dim=1)
                    
                    td_target = rewards + self.gamma * target_max * (1 - dones)
                    
                    predicted_action_values = self.Q_network(states).gather(1, actions.view(-1, 1)).squeeze()
                                        
                    loss = self.loss(td_target, predicted_action_values)
            
                    # optimize the model
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            print("episode: ", self.this_episode, "score: ", score, "epsilon: ", self.epsilon, "learning_rate: ", self.learning_rate)

if __name__ == '__main__':
    agent = QLearningAgent(2, 4, learning_rate_initial=0.0001, epsilon=1, gamma=0.99, env="CartPole-v1", hidden_dim=100, decay_rate=0.001)
    agent.load("test2.pk1")
    try:
        agent.train(episodes=100_000, render=False)
        #agent.train(episodes=500, render=False)
    except KeyboardInterrupt:
        print("Saving!")
        agent.save("test2.pk1")
        pass
    
    agent.save("test2.pk1")
