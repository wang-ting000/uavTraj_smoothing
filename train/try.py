import numpy as np
import matplotlib.pyplot as plt
import math




# r1 = np.load(r'C:\Users\Lenovo\PycharmProjects\uavTraj_smoothing\train\user1\uav\20231024\00\data\user15_dis1000\train\reward\uav_reward.npy',allow_pickle=True)
# r2 = np.load(r'C:\Users\Lenovo\PycharmProjects\uavTraj_smoothing\train\user1\uav\20231024\00\data\user15_dis1000\train\every_reward\reward2\uav_reward2.npy',allow_pickle=True)
# r3 = np.load(r'C:\Users\Lenovo\PycharmProjects\uavTraj_smoothing\train\user1\uav\20231024\00\data\user15_dis1000\train\every_reward\reward3\uav_reward3.npy',allow_pickle=True)
#
# plt.plot(r1,label='r1')
# # plt.plot(r2,label='r2')
# # plt.plot(r3,label='r3')
# plt.legend()
# plt.show()


#
#  #
# payload = np.load(r'C:\Users\Lenovo\PycharmProjects\uavTraj_smoothing\train\user1\uav\20231024\00\data\user15_dis1000\train\payload\iotd_payload.npy',allow_pickle=True)
#
# r = np.load(r'C:\Users\Lenovo\PycharmProjects\uavTraj_smoothing\train\user1\uav\20231024\00\data\user15_dis1000\train\r\R.npy')# #
#
# for i in range(len(payload)):
# for j in range(15):
#     p = [26]
#
#     for i in range(len(r)):
#         p.append(26-min(r[i][j],26))
#     plt.subplot(212)
#     plt.plot(p,label='user%d'%j)
# plt.title('payload with steps after')
# plt.xlabel('step/times')
# plt.ylabel('payload/bps')
# plt.legend()
#
# payload = np.load(r'C:\Users\Lenovo\PycharmProjects\uavTraj_smoothing\train\user\uav\20231024\00\data\user15_dis1000\train\payload\iotd_payload.npy')
#
# for j in range(15):
#     p = [36]
#     for i in range(len(payload)):
#         # print(payload[i])
#         p.append(payload[i][j])
#     plt.subplot(211)
#     plt.plot(p,label='user%d'%j)
#
# plt.title('payload with steps before')
# plt.xlabel('step/times')
# plt.ylabel('payload/bps')
# plt.legend()
# plt.show()
# #
# # print(payload[0])
# print(payload[20])
# print(payload[21])
# print(payload[22])
# print(payload[72])
#
# print('-----')
#
# r = np.load('r1.npy')
# print(r)
# print('------')
# r = np.load('r2.npy')
# print(r)




# com = np.load('remain9.npy',allow_pickle=True)
# print(com)
# #
# com = np.load('payload9.npy',allow_pickle=True)
# print(com)
# for i in range(1,74):
#     print(np.load('choose_num%d.npy'%i,allow_pickle=True))


# print(np.ones((4,2)))

# print(np.load('remain.npy'))
# print(np.load('payload.npy'))
#
# a = np.zeros((73,15))
# a[1:,2] = 9
# print(a)

# a[3:,2:,0] = 9
# print(a)
#
# a[4:,:,0] = 9
# print(a)

#
p_l = np.load('p_l.npy')
# print(p_l)
track = np.load(r'C:\Users\Lenovo\PycharmProjects\uavTraj_smoothing\train\user1\uav\20231024\00\data\user15_dis1000\train\track\uav_track.npy')
tra = np.load(r'C:\Users\Lenovo\PycharmProjects\uavTraj_smoothing\train\user\uav\20231024\00\data\user15_dis1000\train\track\uav_track.npy')


def _genPoints(step_cnt,p_n,p_r):  # 每一段生成midpoints
    oriTraj = np.load(
        r'C:\Users\Lenovo\PycharmProjects\uavTraj_smoothing\train\user\uav\20231024\00\data\user15_dis1000\train\track\uav_track.npy')
    mid1 = [0, 0, 0]
    mid2 = [0, 0, 0]
    mid3 = [0, 0, 0]
    p_n = p_r = 0.3
    if step_cnt == 1:  # 第一段是一阶贝塞尔
        p = oriTraj[0]
        q = oriTraj[1]
        points = np.zeros((3, 2))
        mid1[0] = p_n * q[0] + (1 - p_n) * p[0]
        mid1[1] = p_n * q[1] + (1 - p_n) * p[1]
        mid1[2] = p_n * q[2] + (1 - p_n) * p[2]
        points[0] = [p[0], mid1[0]]
        points[1] = [p[1], mid1[1]]
        points[2] = [p[2], mid1[2]]
        return points
    if step_cnt == len(oriTraj):  # 最后一段是二阶
        p = oriTraj[step_cnt - 2]
        q = oriTraj[step_cnt - 1]
        points = np.zeros((3, 3))
        mid1[0] = p_r * q[0] + (1 - p_r) * p[0]
        mid1[1] = p_r * q[1] + (1 - p_r) * p[1]
        mid1[2] = p_r * q[2] + (1 - p_r) * p[2]
        mid2[0] = p_r * p[0] + (1 - p_r) * q[0]
        mid2[1] = p_r * p[1] + (1 - p_r) * q[1]
        mid2[2] = p_r * p[2] + (1 - p_r) * q[2]
        points[0] = [mid1[0], mid2[0], q[0]]
        points[1] = [mid1[1], mid2[1], q[1]]
        points[2] = [mid1[2], mid2[2], q[2]]
        return points
    else:
        p = oriTraj[step_cnt - 2]
        q = oriTraj[step_cnt - 1]
        r = oriTraj[step_cnt]
        points = np.zeros((3, 4))
        mid1[0] = p_r * q[0] + (1 - p_r) * p[0]
        mid1[1] = p_r * q[1] + (1 - p_r) * p[1]
        mid1[2] = p_r * q[2] + (1 - p_r) * p[2]
        mid2[0] = p_r * p[0] + (1 - p_r) * q[0]
        mid2[1] = p_r * p[1] + (1 - p_r) * q[1]
        mid2[2] = p_r * p[2] + (1 - p_r) * q[2]
        mid3[0] = p_n * r[0] + (1 - p_n) * q[0]
        mid3[1] = p_n * r[1] + (1 - p_n) * q[1]
        mid3[2] = p_n * r[2] + (1 - p_n) * q[2]
        points[0] = [mid1[0], mid2[0], q[0], mid3[0]]
        points[1] = [mid1[1], mid2[1], q[1], mid3[1]]
        points[2] = [mid1[2], mid2[2], q[2], mid3[2]]
    return points




def cal_Bezier(p,step_cnt):
    gap =  10
    def comb(n, m):
        return (math.factorial(n) / (math.factorial(m) * math.factorial(n - m)))

    B = np.zeros((gap, 3))
    st = 0
    for t in np.linspace(0, 1, gap, endpoint=True):
        if step_cnt == 1:
            nn = 1
        elif step_cnt == 94:
            nn = 2
        else:
            nn = 3
        for i in range(nn + 1):
            B[0 + st][0] += comb(nn, i) * p[0][i] * ((1 - t) ** (nn - i)) * (t ** i)
            B[0 + st][1] += comb(nn, i) * p[1][i] * ((1 - t) ** (nn - i)) * (t ** i)
            B[0 + st][2] += comb(nn, i) * p[2][i] * ((1 - t) ** (nn - i)) * (t ** i)
        st += 1
    return B

#
# fig = plt.figure()
# ax1 = plt.axes(projection='3d')
# x = []
# y = []
# z = []
#
# for step_cnt in range(1,len(tra)+1):
#     if step_cnt == 1:
#         points = _genPoints(step_cnt,p_l[step_cnt-1],0)
#     else:
#         points = _genPoints(step_cnt, p_l[step_cnt - 1], p_l[step_cnt-2])
    # print(points)
    # for i in range(len(points[0])):
    #     x.append(points[0][i])
    #     y.append(points[1][i])
    #     z.append(points[2][i])
    # print(tra[step_cnt-1],tra[step_cnt],p_l[step_cnt-1])

#     l = cal_Bezier(points,step_cnt)
#     for i in range(len(l)):
#         x.append(l[i][0])
#         y.append(l[i][1])
#         z.append(l[i][2])
# ax1.plot3D(x,y,z,'k')
#
# #
# x = []
# y = []
# z = []
# for i in range(len(tra)):
#     x.append(tra[i][0])
#     y.append(tra[i][1])
#     z.append(tra[i][2])
# ax1.plot3D(x,y,z,'r')
# plt.show()

def cal_length( line):
    length = 0.0

    for i in range(len(line) - 1):
        x1, y1, z1 = line[i]
        x2, y2, z2 = line[i + 1]

        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        length += distance
    return length

for i in range(0,len(tra)):
    print(cal_length(tra[i:i+2]))
    print(tra[i:i + 2])
    print(cal_length(track[i*10:(i*10 + 10)]))
    print(track[i * 10:(i * 10 + 10)])
    print('-')


print(np.load('midp3.npy'))
print(np.load('midp4.npy'))