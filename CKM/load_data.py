import os
import glob
import numpy as np
# 177.434 37.8373 14.8735
# 185.655 55.2141 1.81542
# 104.199 90.0462 34.8893
# 395.684 130.774 1.56282
# 483.555 64.5507 1.78421
# 517.763 590.942 67.4083
# 671.731 70.1813 45.7654
# 690.378 84.6976 1.71685
# 171.043 565.551 244.464
# 138.551 569.255 -4.28812
# 161.04 579.445 -3.97482
# 940.354 696.339 3.32761
# 965.728 736.252 64.7628
# 632.927 881.275 -3.2105
# 347.106 914.701 0.399067

# with open('Tx_pos','r') as f:
#     lines = f.readlines()
#     N = len(lines) # 无人机轨迹点数
#     print('一共有',N,'个点')
#
#     tx = np.zeros((N,3))
#     index = 0
#     for line in lines:
#         l = line.split()
#         tx[index,:] = l[1:4]
#         index += 1
# np.save('tx.npy',tx)
# print(np.load('tx.npy'))

Tx = np.load(r'D:\A_codes\uavTraj_smoothing\CKM\tx.npy')
N= len(Tx)

path = 'D:/wireless insite/a2garea'
files = os.listdir(path)
pattern = 'myUAV3DScene2_eq.pg.t*_25.r003.p2m'
path_pattern = os.path.join(path, pattern)
inputs = np.zeros((N*15,6))
channel_gains = np.zeros((N*15,1))
index = 0
mins = np.array([0,0,250])
maxs = np.array([1000,1000,750])
Tx_i = 0
# for i in range(len(Tx)):
#     with open('../CKM/UE_q.txt') as f:
#         for j in range(15):
#             l = f.readline().strip('\n').split(' ')
#             inputs[index, :3] = [eval(l[0]),eval(l[1]),eval(l[2])]
#             inputs[index,3:] = Tx[Tx_i]
#             index += 1
#         Tx_i += 1

for file in glob.glob(path_pattern):
    # 遍历每一个无人机轨迹点
    with open(file,'r') as f:
        # 遍历每一个用户
        lines = f.readlines()
        for line in lines[3:]:
            l = line.split(' ')
            # inputs前三个是UE位置，后三个是无人机位置
            inputs[index,:3] = [float(l[i]) for i in range(1,4)]
            inputs[index,3:] = Tx[Tx_i]
            channel_gains[index] = float(l[5])
            index += 1
    Tx_i += 1
# np.save('qs1.npy', inputs)
np.save('channel_gains_new2.npy', channel_gains)









