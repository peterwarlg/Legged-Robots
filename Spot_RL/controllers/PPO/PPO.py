# import gym
import torch
import torch.nn.functional as F
import numpy as np
from controller import Robot, Motor, GPS, Supervisor
# from webots_env import SpotEnv
import rl_utils
import sys
import os

# 获取父目录的父目录路径
grandparent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将其加入sys.path
sys.path.append(grandparent_dir)
# 导入类
from webots_env.spot_env import SpotEnv
import yaml


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x)).clamp_min(1e-7)
        return mu, std


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class PPOContinuous:
    ''' 处理连续动作的PPO算法 '''

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim,
                                         action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, sigma = self.actor(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        # print(action.tolist())
        return action.tolist()[0]

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        rewards = (rewards + 8.0) / 8.0  # 和TRPO一样,对奖励进行修改,方便训练
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        # 动作是正态分布
        old_log_probs = action_dists.log_prob(actions)

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


# class SpotEnv:
#     def __init__(self, TIME_STEP):
#         self.max_step: int = 100
#         self.counter_step: int = 0
#         self.state_dim = 11
#         self.action_dim = 8
#         self.sim_time_step = TIME_STEP
#         self.supervisor = Supervisor()
#         self.gps = self.supervisor.getDevice('gps')
#         self.gps.enable(self.sim_time_step)
#         self.supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)
#
#         self.motor_names = [
#             "front left shoulder rotation motor", "front left elbow motor",
#             "front right shoulder rotation motor", "front right elbow motor",
#             "rear left shoulder rotation motor", "rear left elbow motor",
#             "rear right shoulder rotation motor", "rear right elbow motor"
#         ]
#         self.motors = [self.supervisor.getDevice(name) for name in self.motor_names]
#         for motor in self.motors:
#             motor.setPosition(0.0)  # Initialize to zero position
#             motor.setVelocity(0.0)  # Set velocity
#             motor.getPositionSensor().enable(self.sim_time_step)
#
#     def compute_reward(self) -> float:
#         reward = 0
#         return reward
#
#     def get_observations(self):
#         gps_values = self.gps.getValues()
#         motor_positions = [motor.getPositionSensor().getValue() for motor in self.motors]
#         observations = np.concatenate([gps_values, motor_positions])
#         # return torch.tensor(observations, dtype=torch.float32).to(device)
#         return observations
#
#     def is_done(self) -> bool:
#         if self.counter_step >= self.max_step:
#             return True
#         return False
#
#     def step(self, action):
#         self.counter_step += 1
#         for i, motor in enumerate(self.motors):
#             motor.setPosition(action[i])
#             motor.setVelocity(10)
#         self.supervisor.step(self.sim_time_step * 10)
#
#         next_state = self.get_observations()
#         reward = self.compute_reward()
#         done = self.is_done()
#         return next_state, reward, done, None
#
#     def reset(self):
#         self.counter_step = 0
#         print("Reset called")
#         for motor in self.motors:
#             motor.setPosition(0.0)  # Reset motor positions
#             motor.setVelocity(0.0)
#         self.supervisor.simulationReset()
#         self.supervisor.simulationResetPhysics()  # Reset the physics to apply changes
#         self.supervisor.step(self.sim_time_step * 10)  # Step to apply the reset
#         return self.get_observations()


actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 200
hidden_dim = 256
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env = SpotEnv()

torch.manual_seed(0)
state_dim = env.state_dim
action_dim = env.action_dim
agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                      epochs, eps, gamma, device)

return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)
