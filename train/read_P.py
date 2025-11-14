import glob
import pickle

import numpy as np
from matplotlib import pyplot as plt

# g = np.load('../CKM/channel_gains.npy')
#
#
# for i in range(len(g)):
#     g[i] += 50 if i%15 == 3 else 0
# plt.plot([g[i] for i in range(len(g)) if i%15==3])
# plt.legend()
# plt.show()
#
# np.save('../CKM/channel_gains_adjust.npy',g)

# plt.plot(np.load('pre_noise11.npy'))
# plt.show()
# gains_min, gains_max = np.load(r'D:\A_codes\uavTraj_smoothing\CKM\global_norm.npy')
# g = np.load('../CKM/y_train.npy')
# g = g * (gains_max - gains_min) + gains_min
# plt.plot(g[0:300])
#
# plt.show()
#
# file_patern = 'pre/pre_noise1_*.npz'
# files = glob.glob(file_patern)
# p = []
# x = []
# y = []
# z = []
# pr = []
# for file in files:
#     a = np.load(file,allow_pickle=True)
#     pos = a['inputs']
#     pr.append(a['iotd_received_power_pred_n'])
#
#     # print(a['inputs'])
#     x.append(pos[0][3]*1000)
#     y.append(pos[0][4]*1000)
#     z.append(pos[0][5]*750)
#     print(a['x'],a['y'],a['z'],a['v'],a['theta'],a['phi'],a['com'],a['uav_com'])
#     # print(a['iotd_received_power_pred_n'])
#     # print(a['choose_iotd_power'])
#     # print(a['inputs'])
#     # p.append(a['choose_iotd_power'])
# # print(pr)
# # fig = plt.figure()
# # fig.suptitle("The trajectory of UAV-BS")
# # ax = fig.add_subplot(121, projection="3d")
# # ax.plot(x,y,z)
# # ax.set_xlim(0, 1000)
# # ax.set_ylim(0, 1000)
# # ax.set_zlim(0, 750)
# # plt.show()
# #
# # for j in range(len(pr)):
# #     plt.plot([p for p in pr[j]],marker='*')
# # # plt.plot(p,'g-o')
# # plt.show()

# path/to/your/script.py
# 该脚本读取一个 txt 文件，提取每个循环的前三行，并保存到一个新的 txt 文件中

print(np.load(r''))