import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from controller import Robot, Motor, GPS, Supervisor
import tqdm
from tqdm import trange
import os
import matplotlib.pyplot as plt
#import subprocess

# Constants
TIME_STEP = 32
MAX_EPISODE_STEPS = 120
TARGET_GOAL = [1.2, -0.199, 0.624]  
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.005
MAX_SPEED = 6.28
seed = 42
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)

# Define the Actor and Critic Networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))  
        
        #Motor ranges
        shoulder_rotation_min, shoulder_rotation_max = -1.65, 1.65
        elbow_min, elbow_max = -0.44, 1.55

        shoulder_rotation_range = (shoulder_rotation_max - shoulder_rotation_min) / 2
        shoulder_rotation_mid = (shoulder_rotation_max + shoulder_rotation_min) / 2
        elbow_range = (elbow_max - elbow_min) / 2
        elbow_mid = (elbow_max + elbow_min) / 2

        # Scale values for each motor type without in-place modification
        x_scaled = x.clone()
        x_scaled[:, [0, 2, 4, 6]] =  x[:, [0, 2, 4, 6]] * shoulder_rotation_range + shoulder_rotation_mid
        x_scaled[:, [1, 3, 5, 7]] = x[:, [1, 3, 5, 7]] * elbow_range + elbow_mid

        return x_scaled

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)
    
    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        # #Update the noise process
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x  # Update x_prev to the new state
        return x

    def reset(self):
        # #Initialize x_prev to x0 if provided, otherwise to a zero vector of the same shape as mu
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

        
# DDPG Agent Class
class DDPGAgent:
    def __init__(self):
        self.supervisor = Supervisor()
        self.gps = self.supervisor.getDevice('gps')
        self.gps.enable(TIME_STEP)
        self.supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)

        self.motor_names = [
             "front left shoulder rotation motor", "front left elbow motor",
             "front right shoulder rotation motor", "front right elbow motor",
             "rear left shoulder rotation motor", "rear left elbow motor",
             "rear right shoulder rotation motor", "rear right elbow motor"
        ]

        self.motors = [self.supervisor.getDevice(name) for name in self.motor_names]
        
        for motor in self.motors:
            motor.setPosition(0.0)  # Initialize to zero position
            motor.setVelocity(0.0)  # Set velocity
            motor.getPositionSensor().enable(TIME_STEP)
            
        self.state_dim = 11  
        self.action_dim = 8  

        self.actor = Actor(self.state_dim, self.action_dim).cuda()
        self.critic = Critic(self.state_dim, self.action_dim).cuda()
        self.target_actor = Actor(self.state_dim, self.action_dim).cuda()
        self.target_critic = Critic(self.state_dim, self.action_dim).cuda()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # Copy the weights to the target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.memory = deque(maxlen=1000000)  # Replay buffer
        self.save_interval = 50
        self.episode_rewards = []
        self.ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_dim))
        self.previous_distance = None
        
    def get_observations(self):
        gps_values = self.gps.getValues()
        motor_positions = [motor.getPositionSensor().getValue() for motor in self.motors]
        observations = np.concatenate([gps_values, motor_positions])
        return torch.tensor(observations, dtype=torch.float32).cuda()

    def compute_reward(self):
        gps_values = self.gps.getValues()
        distance_to_goal = np.linalg.norm(np.array(TARGET_GOAL) - np.array(gps_values))
        reward = -distance_to_goal  # Reward is negative distance to goal
        
        # Stability reward
        stable_height = 0.6  
        height_tolerance = 0.1  
    
        if abs(gps_values[2] - stable_height) < height_tolerance:
            reward += 2  
            
        # Progress reward
        if self.previous_distance is not None:
            progress = self.previous_distance - distance_to_goal
            reward += progress * 10  
    
        self.previous_distance = distance_to_goal
    
        if distance_to_goal < 0.1:  # If within 10 cm of the target goal
            reward += 1000  # Large reward for reaching the goal
        return reward

    def reset(self):
        print("Reset called")
        for motor in self.motors:
            motor.setPosition(0.0)  # Reset motor positions
            motor.setVelocity(0.0)
        self.supervisor.simulationReset()
        self.supervisor.simulationResetPhysics()  # Reset the physics to apply changes
        self.supervisor.step(TIME_STEP*10)  # Step to apply the reset

    def save_checkpoint(self, filepath):
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'replay_buffer': self.memory,
            'episode_rewards': self.episode_rewards
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.memory = checkpoint['replay_buffer']
        self.episode_rewards = checkpoint["episode_rewards"]
        print(f"Checkpoint loaded from {filepath}")

    def train(self, episodes=1000):
        for episode in range(1,episodes+1):
            self.reset()
            state = self.get_observations()
            episode_reward = 0
            self.ou_noise.reset()
            self.previous_distance = None

            for step in range(MAX_EPISODE_STEPS):
                with torch.no_grad():
                    action = self.actor(state.unsqueeze(0)).cpu().detach().numpy()
                    action = action.squeeze(0)
                    noise = self.ou_noise()  
                    action += noise  
                    action[[0, 2, 4, 6]] = np.clip(action[[0, 2, 4, 6]],-1.65, 1.65)
                    action[[1, 3, 5, 7]] = np.clip(action[[1, 3, 5, 7]],-0.44, 1.55)
                    for i, motor in enumerate(self.motors):
                        motor.setPosition(action[i])
                        motor.setVelocity(MAX_SPEED)

                self.supervisor.step(TIME_STEP*10)
                next_state = self.get_observations()
                reward = self.compute_reward()
                done = reward > 900 or step == MAX_EPISODE_STEPS - 1

                self.memory.append((state, action, reward, next_state, done))
                if len(self.memory) > 1000:
                    batch = random.sample(self.memory, BATCH_SIZE)
                    states, actions, rewards, next_states, dones = zip(*batch)

                    states = torch.stack(states).cuda()
                    actions = torch.tensor(np.array(actions), dtype=torch.float32).cuda()
                    rewards = torch.tensor(np.array(rewards), dtype=torch.float32).cuda()
                    next_states = torch.stack(next_states).cuda()
                    dones = torch.tensor(np.array(dones), dtype=torch.float32).cuda()
                    
                    with torch.autograd.set_detect_anomaly(True):
                        # Critic loss
                        target_q_values = self.target_critic(next_states, self.target_actor(next_states))
                        target_q_values = target_q_values.squeeze()
                        expected_q_values = rewards + (1 - dones) * GAMMA * target_q_values
                        q_values = self.critic(states, actions)
                        expected_q_values = expected_q_values.unsqueeze(1)
                        critic_loss = nn.MSELoss()(q_values, expected_q_values.detach()) #
                        self.critic_optimizer.zero_grad()
                        critic_loss.backward()
                        self.critic_optimizer.step()
    
                        # Actor loss
                        predicted_actions = self.actor(states)
                        actor_loss = -self.critic(states, predicted_actions).mean()
                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer.step()
    
                        # Soft update of target networks
                        self.soft_update(self.actor, self.target_actor)
                        self.soft_update(self.critic, self.target_critic)
                    
                state = next_state
                episode_reward += reward
                if done:
                    print("Episode done...")
                    self.episode_rewards.append(episode_reward)
                    break

            print(f"Episode {episode}, Reward: {episode_reward}")
            if episode%self.save_interval==0:
                self.save_checkpoint(f"checkpoint_{episode}.pth")

    def soft_update(self, model, target_model):
        with torch.no_grad():
            for weights, target_weights in zip(model.parameters(), target_model.parameters()):
                new_data_param = TAU * weights.data + (1 - TAU) * target_weights.data
                target_weights.data.copy_(new_data_param)
                
    def test(self):
        self.reset()
        state = self.get_observations()
        episode_reward = 0

        for step in range(MAX_EPISODE_STEPS):
            with torch.no_grad():
                        action = self.actor(state.unsqueeze(0)).cpu().detach().numpy()
                        action = action.squeeze(0)
                        for i, motor in enumerate(self.motors):
                            motor.setPosition(action[i])
                            motor.setVelocity(MAX_SPEED)
            self.supervisor.step(TIME_STEP*10)
            state = self.get_observations()
            reward = self.compute_reward()
            episode_reward += reward

            if reward > 900:  # If the robot reaches the target goal
                print(f"Reached goal in {step} steps!")
                break

        print(f"Test Reward: {episode_reward}")
        
    def plot_rewards(self, rewards: list) -> None:
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(rewards_t.numpy()) 

if __name__ == "__main__":
    agent = DDPGAgent()
    print("DDPG agent created")
    train_mode = True    
    #agent.load_checkpoint("checkpoint_100.pth")
    # buffer_ckpt = torch.load("checkpoint_replay_2.pth")
    
    # print("Memory : " , len(agent.memory))
    # print("Replay buffer : ", len(buffer_ckpt["replay_buffer"]))
    # for i in range(1):
        # for transition in buffer_ckpt["replay_buffer"]:
            # agent.memory.append(transition)
    # print("Memory After: " , len(agent.memory))
    if train_mode:
        agent.train(episodes=200)
    else:
        agent.test()
    agent.plot_rewards(agent.episode_rewards)
    plt.show()
