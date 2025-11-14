import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

payload = np.load(r'D:\A_codes\uavTraj_smoothing\evaluate\plot\payload.npy')
power = np.load(r'D:\A_codes\uavTraj_smoothing\evaluate\plot\power.npy')
# payload = payload.T
# payload = np.load(r'C:\Users\Lenovo\PycharmProjects\uavTraj_smoothing\train\user1\uav\20231208\00\data\user15_dis1000\train\payload\iotd_payload.npy')
marker = ['$\clubsuit$', 'o', 'v', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'x', 'D', 'd', '+']
fig = plt.figure(figsize=(16,12))
plt.subplot(211)
plt.title('(a)')
for i in range(15):
    plt.plot(payload[:,i], label='iotd%s' % i, marker=marker[i], markersize=8, alpha=0.6,linewidth=2.5)
plt.plot(power,'r^',label='power')
# plt.title('communication status of TST-PPO',fontsize=30)
plt.xlabel('t/s',fontsize=16)
plt.ylabel('payload',fontsize=16)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.grid()
plt.legend(fontsize=10)
##################################################################################################################
plt.subplot(212)
payload = np.load(r'D:\A_codes\uavTraj_smoothing\evaluate\plot\payload1.npy')
power = np.load(r'D:\A_codes\uavTraj_smoothing\evaluate\plot\power1.npy')
for i in range(15):
    plt.plot(payload[:,i], label='iotd%s' % i, marker=marker[i], markersize=8, alpha=0.6,linewidth=2.5)
plt.plot(power,'r^',label='power')
# plt.title('communication status of TST-PPO',fontsize=30)
plt.xlabel('t/s',fontsize=16)
plt.ylabel('payload',fontsize=16)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.grid()
plt.title('(b)')
plt.legend(fontsize=10)
plt.subplot(212)

plt.savefig('OS.png',dpi=300,bbox_inches='tight')
plt.show()


