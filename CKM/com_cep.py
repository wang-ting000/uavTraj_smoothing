import matplotlib.pyplot as plt
import numpy as np
from uav_gym.envs.uav_env import UavConfig

top_config = UavConfig(train_flag=2)
iotd_pos = np.load(r'D:\A_codes\uavTraj_smoothing\iotd_pos.npy')
track1 = np.load(r'D:\A_codes\uavTraj_smoothing\evaluate\plot\pos.npy')
f2 = open(r'D:\A_codes\uavTraj_smoothing\evaluate\user\uav\20240521\00\data\user15_dis1000\evaluate_log_file\evaluate_file.txt')
# f3 = open(r'D:\A_codes\uavTraj_smoothing\evaluate\user\uav\20240521\01\data\user15_dis1000\evaluate_log_file\evaluate_file.txt')
f4 = open(r'D:\A_codes\uavTraj_smoothing\evaluate\user\uav\20240521\02\data\user15_dis1000\evaluate_log_file\evaluate_file.txt')
f3 = open(r'D:\A_codes\uavTraj_smoothing\evaluate\user\uav\20240521\01\data\user15_dis1000\evaluate_log_file\evaluate_file.txt')
dict = {'f2':f2,'f3':f3,'f4':f4}
d2 = {'l2':'l2','l3':'l3','l4':'l4'}
for i in range(2,5):
    varname = 'f'+str(i)
    f = dict[varname]
    line = f.readline()
    data_list = []
    while line:
        num = str(list(map(float, line.split()))).strip('[').strip(']' )
        num = float(num)
        data_list.append(num)
        line = f.readline()
    f.close()
    length = np.shape(data_list)[0]
    t = int((length - 1) / top_config.STATE_DIM)
    var = 'l' + str(i)
    d2[var] = str(t) + 's'
    pos = np.zeros((t + 1, 3))
    for k in range(t + 1):
        pos[k][0] = data_list[k * top_config.STATE_DIM + 0] * top_config.total_x
        pos[k][1] = data_list[k * top_config.STATE_DIM + 1] * top_config.total_y
        pos[k][2] = data_list[k * top_config.STATE_DIM + 2] * top_config.h_max
    np.save('track'+str(i),pos)

track2 = np.load('track2.npy')
track3 = np.load('track3.npy')
track4 = np.load('track4.npy')
labels = ['CEP=0, 35s','CEP=1m, 40s', 'CEP=5m, 44s', 'CEP=10m, 47s']
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111,projection='3d')
ax.view_init(90, 45)
ax.set_xlabel('X(m)')
ax.set_ylabel('Y(m)')
ax.set_zlabel('Z(m)')
ax.scatter(top_config.UAV_INITIAL_POSITION_X, top_config.UAV_INITIAL_POSITION_Y,
                   top_config.UAV_INITIAL_POSITION_Z, c='r', marker='x',
                   s=30)
ax.set_xlim(0, top_config.total_x)
ax.set_ylim(0, top_config.total_y)
ax.set_zlim(0, top_config.h_max)
for i in range(15):
    ax.scatter(iotd_pos[i,0], iotd_pos[i,1],iotd_pos[i,2], c=iotd_pos[i,2], marker='o',s=30)
ax.plot(track1[:,0], track1[:,1],track1[:,2],'k--',linewidth=2,label=labels[0])
ax.plot(track2[:,0], track2[:,1],track2[:,2],'g-+',linewidth=2,label=labels[1])
ax.plot(track3[:,0], track3[:,1],track3[:,2],'m-*',linewidth=2,label=labels[2])
ax.plot(track4[:,0], track4[:,1],track4[:,2],'p-.',linewidth=2,label=labels[3])
# plt.legend(loc='upper right')
# plt.title('influences of CEP for UAV trajectory design')
# plt.savefig('CEP对比_俯视图.png',dpi=3000, bbox_inches='tight')
plt.show()


