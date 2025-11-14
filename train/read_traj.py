import numpy as np

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

trj = np.load(r'D:\A_codes\uavTraj_smoothing\train\user\uav\20231130\00\data\user15_dis1000\train\track\uav_track.npy')

print(trj.shape)
x = trj[:,0]
y = trj[:,1]
z = trj[:,2]
ax.scatter(x, y,z)
plt.show()