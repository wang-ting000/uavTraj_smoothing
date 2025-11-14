import os.path

import numpy as np
import matplotlib.pyplot as plt

r = np.load(r'C:\Users\Lenovo\PycharmProjects\uavTraj_smoothing\train\user1\uav\20231208\00\data\user15_dis1000\train\every_reward\reward2\uav_reward2.npy')
plt.plot(r)
plt.show()
plt.figure()
r = np.load(r'C:\Users\Lenovo\PycharmProjects\uavTraj_smoothing\train\user1\uav\20231208\00\data\user15_dis1000\train\every_reward\reward3\uav_reward3.npy')
plt.plot(r)
plt.show()