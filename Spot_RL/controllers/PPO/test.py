import torch
import torch.nn.functional as F
import numpy as np
# from controller import Robot, Motor, GPS, Supervisor
import rl_utils
import sys

sys.path.append(r"..\..\envs")

from webots_env.spot_env import SpotEnv

if __name__ == '__main__':
    spot = SpotEnv()