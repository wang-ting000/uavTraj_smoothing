import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from train.KDML_CKM import calculate_path_loss
#
r1 = np.load(r'D:\A_codes\uavTraj_smoothing\train\user\uav\20240920\00\data\user15_dis1000\train\every_reward\reward1\uav_reward1.npy')
plt.plot(r1)
plt.show()
r1 = np.load(r'D:\A_codes\uavTraj_smoothing\train\user\uav\20240920\00\data\user15_dis1000\train\every_reward\reward2\uav_reward2.npy')
plt.plot(r1)
plt.show()
r1 = np.load(r'D:\A_codes\uavTraj_smoothing\train\user\uav\20240920\00\data\user15_dis1000\train\every_reward\reward3\uav_reward3.npy')
plt.plot(r1)
plt.show()
r1 = np.load(r'D:\A_codes\uavTraj_smoothing\train\user\uav\20240920\00\data\user15_dis1000\train\every_reward\reward4\uav_reward4.npy')
plt.plot(r1)
plt.show()
print(np.load(r'D:\A_codes\uavTraj_smoothing\train\user\uav\20240920\00\data\user15_dis1000\train\power\uav_iotd_com_power.npy'))
print(np.load(r'D:\A_codes\uavTraj_smoothing\train\user\uav\20240920\00\data\user15_dis1000\train\track\uav_track.npy'))
print(np.load(r'D:\A_codes\uavTraj_smoothing\train\user\uav\20240920\00\data\user15_dis1000\train\r\R.npy'))
# GU = np.load(r'D:\A_codes\uavTraj_smoothing\train\train\pos\iotd_path.npy')
# uav_x_pos = 150
# uav_y_pos = 750
# uav_z_pos = 750
# inputs = np.hstack((GU, np.tile(np.array([uav_x_pos, uav_y_pos, uav_z_pos]),
#                                                    (np.array(GU).shape[0], 1))))
# model = load_model(r'D:\A_codes\Trans2024\CKM\kdml_model_with_knowledge_in_hidden_layer_augmented.h5')
# env_now = np.load('CKM/environment_scaled.npy')[0:15]
# pred_channelgain = model.predict([inputs,env_now], verbose=0) # inputs不归一化
# [gains_min,gains_max] = np.load(r'D:\A_codes\Trans2024\CKM\global_norm_augmented.npy')
#
# def calculate_distance_and_theta(user_pos, uav_pos):
#     """
#     计算 UAV 和用户之间的实际距离和仰角.
#     user_pos: 用户的 (x, y, z) 坐标
#     uav_pos: 无人机的 (x, y, z) 坐标
#     """
#     # 计算水平距离
#     horizontal_distance = np.sqrt(np.sum((user_pos[:2] - uav_pos[:2]) ** 2)) + 1e-6  # 防止除0
#     # 计算实际距离
#     real_distance = np.sqrt(horizontal_distance ** 2 + (user_pos[2] - uav_pos[2]) ** 2)
#     # 计算仰角 (radian)
#     theta = np.arctan((uav_pos[2] - user_pos[2]) / horizontal_distance)
#
#     return real_distance, theta
#
#
# # 计算每一行中 UAV 和用户之间的路径损耗
# def calculate_path_loss_for_8dim_array(array_8dim, A, a, b, B):
#     """
#     array_8dim: 一个 numpy 数组, 其中前6个维度是用户和 UAV 的坐标
#     A, a, b, B: 路径损耗计算中的参数
#     """
#     path_loss_list = []
#     for row in array_8dim:
#         user_pos = row[:3]  # 用户的 (x, y, z) 坐标
#         uav_pos = row[3:6]  # UAV 的 (x, y, z) 坐标
#
#         # 计算距离和仰角
#         real_distance, theta_rad = calculate_distance_and_theta(user_pos, uav_pos)
#
#         # 将仰角从弧度转换为角度
#         theta_deg = 180 / np.pi * theta_rad
#
#         # 计算路径损耗的第一部分（与仰角相关）
#         path_loss_angle_part = A / (1 + a * np.exp(-b * (theta_deg - a)))
#
#         # 计算路径损耗的第二部分（与距离相关）
#         path_loss_distance_part = 20 * np.log10(real_distance)
#
#         # 总路径损耗
#         path_loss = path_loss_angle_part + path_loss_distance_part + B
#         path_loss_list.append(-path_loss)
#
#     return np.array(path_loss_list)
#
# los = calculate_path_loss_for_8dim_array(inputs, 1, 9.61, 0.16, 20)
# pred_channelgain = pred_channelgain*(gains_max-gains_min)+gains_min
# plt.plot(pred_channelgain)
# plt.plot(np.ones(pred_channelgain.shape)*(-106))
# plt.plot(los,'r')
# plt.show()