import numpy as np

import matplotlib.pyplot as plt
import math

def cal_length( line):
    length = 0.0

    for i in range(len(line) - 1):
        x1, y1, z1 = line[i]
        x2, y2, z2 = line[i + 1]

        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        length += distance
    return length

track = np.load(r'C:\Users\Lenovo\PycharmProjects\uavTraj_smoothing\evaluate\plot\pos.npy')
#
x = np.zeros(len(track))
y = np.zeros(len(track))
z = np.zeros(len(track))

for i in range(len(track)):
    x[i] = track[i][0]
    y[i] = track[i][1]
    z[i] = track[i][2]

pos = np.load(r'C:\Users\Lenovo\PycharmProjects\uavTraj_smoothing\evaluate\plot\iotd_pos.npy')

xu = np.zeros(len(pos))
yu = np.zeros(len(pos))
zu = np.zeros(len(pos))

for i in range(len(pos)):
    xu[i] = pos[i][0]
    yu[i] = pos[i][1]
    zu[i] = pos[i][2]

track1 = np.load(r'C:\Users\Lenovo\PycharmProjects\uavTraj_smoothing\train\user1\uav\20231130\00\data\user15_dis1000\train\track\uav_track.npy')
#
print(track1)
print(track)
x1 = np.zeros(len(track1))
y1 = np.zeros(len(track1))
z1 = np.zeros(len(track1))

for i in range(len(track1)):
    x1[i] = track1[i][0]
    y1[i] = track1[i][1]
    z1[i] = track1[i][2]
from mpl_toolkits.mplot3d import Axes3D

# 定义坐标轴
fig = plt.figure()
ax1 = plt.axes(projection='3d')
ax1.plot3D(x,y,z,'grey',label='trajectory of PPO')
ax1.plot3D(x1,y1,z1,'green',label='trajectory of TST-PPO')
ax1.scatter3D(xu,yu,zu,'red',s=50,label='UEs')
plt.legend(fontsize=16)
plt.title('trajectory comparation',fontsize=16)
plt.show()





