from gym.envs.registration import register
import numpy as np

import uav_gym.envs

register(
    id='uav_env-v0',                                   # Format should be xxx-v0, xxx-v1....
    entry_point='uav_gym.envs.uav_env:uavEnv',              # Expalined in envs/__init__.py
    max_episode_steps=160,
)

t = np.load(r'C:\Users\Lenovo\PycharmProjects\uavTraj_smoothing\evaluate\plot\pos.npy')

register(
    id='uav_env1-v0',                                   # Format should be xxx-v0, xxx-v1....
    entry_point='uav_gym.envs:uavEnv1',              # Expalined in envs/__init__.py
    max_episode_steps=len(t),
)
