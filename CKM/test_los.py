import numpy as np
from matplotlib import pyplot as plt

# LOS = 1
# NLOS = 20
# A = LOS - NLOS
#         #
# c = 3 * (10 ** 8)
# # noise
# a = 9.61
# b = 0.16
# fc = 2000 * (10 ** 6)
# B = 20 * np.log10(4 * np.pi * fc / c) + NLOS
#
# pg = np.load('channel_gains.npy')
# Tx = np.load('tx.npy')
#
# iotd_pos = np.zeros((15,3))
# with open('UE_q.txt') as f:
#     for i in range(15):
#         l = f.readline().strip('\n').split(' ')
#         iotd_pos[i] = ([eval(l[0]),eval(l[1]),eval(l[2])])
#
# N = len(pg)//15
# uav_x_pos = Tx[:,0]
# uav_y_pos = Tx[:,1]
# uav_z_pos = Tx[:,2]
#
# uav_iotd_horizontal_distance = np.zeros(15*N)
# uav_iotd_real_distance = np.zeros(15*N)
# uav_iotd_theta = np.zeros(15*N)
# uav_iotd_path_loss = np.zeros(15*N)
# for j in range(N):
#     for i in range(15):
#         uav_iotd_horizontal_distance[j*15+i] = np.sqrt(np.square(uav_x_pos[j] - iotd_pos[i][0])
#                                                        + np.square(uav_y_pos[j] - iotd_pos[i][1])) + 1
#
#
#
#         uav_iotd_real_distance[j*15+i] = np.sqrt(np.square(uav_iotd_horizontal_distance[i])
#                                                  + np.square(uav_z_pos[j] - iotd_pos[i][2]))
#
#
#
#         uav_iotd_theta[j*15+i] = np.arctan(
#             np.true_divide(uav_z_pos[j] - iotd_pos[i][2], uav_iotd_horizontal_distance[j*15+i]))
#
#
#         uav_iotd_path_loss[j*15+i] = -(np.true_divide(A, 1 + a * np.exp(
#             -b * (180 / np.pi * uav_iotd_theta[j*15+i] - a))) + 20 * np.log10(
#             uav_iotd_real_distance[j*15+i]) + B)
#
# plt.plot(uav_iotd_path_loss,'b')
#
# plt.plot(pg,'r')
# plt.figure()
# plt.plot(pg,'r')
# plt.plot(uav_iotd_path_loss,'b')
#
#
#
# plt.show()
#
# np.save('los_gain',uav_iotd_path_loss)

iotd_pos = np.zeros((15, 3))
with open('UE_q.txt') as f:
    for i in range(15):
        l = f.readline().strip('\n').split(' ')
        iotd_pos[i] = ([eval(l[0]), eval(l[1]), eval(l[2])])
Tx = np.load('tx.npy')
iotd_pos = np.tile(iotd_pos, (100, 1))

new_inputs = np.hstack((iotd_pos, Tx[0:1500]))