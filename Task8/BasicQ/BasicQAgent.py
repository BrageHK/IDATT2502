import numpy as np
import gymnasium as gym
from collections import defaultdict
import pickle 

class QLearningAgent():
    def __init__(self, action_dim, observed_dim, learning_rate_initial, epsilon, gamma, env):
        self.action_dim = action_dim
        self.observed_dim = observed_dim
        self.learning_rate_initial = learning_rate_initial
        self.learning_rate = learning_rate_initial
        self.epsilon = epsilon
        self.epsilon_initial = epsilon
        self.gamma = gamma
                
        self.env = env
        self.this_episode = 0
        self.decay_rate = 0.0005
        self.Q_values = defaultdict(lambda: np.zeros(self.action_dim)) # dictionary der key er state og verdien er en liste av lengde action_dim

    def update_Q_values(self, state, new_state, reward, action):
        self.Q_values[state][action] += self.learning_rate * (reward + self.gamma * np.max(self.Q_values[new_state] - self.Q_values[state][action]))
        
    def decay_learning_rate(self):
        self.learning_rate = self.learning_rate_initial * 1/(1 + self.decay_rate * self.this_episode)
    
    def decay_epsilon(self):
        self.epsilon = self.epsilon_initial * 1/(1 + self.decay_rate * self.this_episode)
        
    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            return np.argmax(self.Q_values[state])
        
    def values_to_file(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(dict(self.Q_values), file)

    def read_q_values(self, filename):
        # Using pickle.load to retrieve Q_values
        with open(filename, 'rb') as file:  # 'rb' stands for "read binary"
            loaded_data = pickle.load(file)
            self.Q_values = defaultdict(lambda: np.zeros(self.action_dim), loaded_data)  # Convert dict back to defaultdict

                
    def observation_to_tuple(self, observation):
        return tuple(np.round(observation, 1)) # Runder av til 1 desimal for jeg vil ikke trene på så små forskjeller
        
    def train(self, episodes, render = False, write_to_file=True):
        env_render = gym.make(self.env, render_mode="human")
        env_no_render = gym.make(self.env)
        for _ in range(episodes):
            env = env_no_render
            if(render):
                env = env_render
            observation, info = env.reset()
            
            self.this_episode += 1

            self.decay_learning_rate()
            self.decay_epsilon()
            
            prev_state = self.observation_to_tuple(observation)
            action = self.act(prev_state)
            terminated = False
            truncated = False
            score = 0
            while(not (terminated or truncated)):
                observation, reward, terminated, truncated, info = env.step(action)
                score += reward
                self.update_Q_values(prev_state, self.observation_to_tuple(observation), reward, action)
                action = self.act(self.observation_to_tuple(observation))
                prev_state = self.observation_to_tuple(observation)
            print("episode: ", self.this_episode, "score: ", score, "epsilon: ", self.epsilon, "learning_rate: ", self.learning_rate)
        
        if(write_to_file):
            self.values_to_file("values.pk1")
            


if __name__ == '__main__':
    agent = QLearningAgent(2, 4, learning_rate_initial=0.1, epsilon=0., gamma=0.9, env="CartPole-v1")
    
    agent.read_q_values("values.pk1") # Leser Q_values fra fil
    agent.train(episodes=10000, render=True) # Trener agenten
    
    agent.train(episodes=1, render=True, write_to_file=False) # visualisering
