import os

import numpy as np
import matplotlib.pyplot as plt

iotd_num = 15
iotd_x_pos = np.zeros(iotd_num)
iotd_y_pos = np.zeros(iotd_num)
iotd_z_pos = np.zeros(iotd_num)
iotd_pos = []
with open('CKM/UE_q.txt') as f:
    for i in range(iotd_num):
        l = f.readline().strip('\n').split(' ')
        print(l)
        iotd_x_pos[i] = eval(l[0])
        iotd_y_pos[i] = eval(l[1])
        iotd_z_pos[i] = eval(l[2])
        iotd_pos.append([iotd_x_pos[i], iotd_y_pos[i], iotd_z_pos[i]])
    iotd_pos_path = 'train\pos'
    if not os.path.exists(iotd_pos_path):
        os.makedirs(iotd_pos_path)

np.save('iotd_pos.npy', iotd_pos)