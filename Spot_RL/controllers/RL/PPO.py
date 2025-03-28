# import gym
import torch
import torch.nn.functional as F
import numpy as np
from controllers.PPO import rl_utils
from controller import Supervisor


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    ''' PPO算法,采用截断方式 '''

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


class SpotEnv:
    def __init__(self, TIME_STEP):
        self.max_step: int = 100
        self.counter_step: int = 0
        self.state_dim = 11
        self.action_dim = 8
        self.time_step = TIME_STEP
        self.supervisor = Supervisor()
        self.gps = self.supervisor.getDevice('gps')
        self.gps.enable(self.time_step)
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
            motor.getPositionSensor().enable(self.time_step)

    def compute_reward(self) -> float:
        reward = 0
        return reward

    def get_observations(self):
        gps_values = self.gps.getValues()
        motor_positions = [motor.getPositionSensor().getValue() for motor in self.motors]
        observations = np.concatenate([gps_values, motor_positions])
        return torch.tensor(observations, dtype=torch.float32).to(device)

    def is_done(self) -> bool:
        if self.counter == self.max_step:
            return True
        return False

    def step(self, action):
        self.counter_step += 1
        for i, motor in enumerate(self.motors):
            motor.setTorque(action[i])
        self.supervisor.step(self.time_step * 10)

        next_state = self.get_observations()
        reward = self.compute_reward()
        done = self.is_done()
        return next_state, reward, done, None

    def reset(self):
        self.counter_step = 0
        print("Reset called")
        for motor in self.motors:
            motor.setPosition(0.0)  # Reset motor positions
            motor.setVelocity(0.0)
        self.supervisor.simulationReset()
        self.supervisor.simulationResetPhysics()  # Reset the physics to apply changes
        self.supervisor.step(self.time_step * 10)  # Step to apply the reset


actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 500
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env = SpotEnv(32)

torch.manual_seed(0)
state_dim = env.state_dim
action_dim = env.action_dim
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma, device)

return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)
