import os.path

import numpy as np
import matplotlib.pyplot as plt
from uav_gym.envs.uav_env import UavConfig

###########---------------------##########
##本函数用于画出一定回合数的奖励函数###
##train_flag为true画train过程中的值，false则画evaluate的值
def smooth_curve(l, data, x):
    new_data = [0] * l
    for i in range(l):
        new_data[i] = np.average(data[i:i + x])
    return new_data


def plot_reward(reward_file, top_config, file_path, order):
    reward_read = np.load(reward_file)
    show_reward_read = []
    for i in range(len(reward_read)):
        if i == 0:
            show_reward_read.append(reward_read[i])
        else:
            show_reward_read.append(show_reward_read[i - 1] * 0.99 + 0.01 * reward_read[i])
            # 需要与train_ppo的gama一致
    plt.plot(np.arange(len(show_reward_read)), show_reward_read, 'purple', alpha=0.6, label='reward')
    plt.xlabel('Episode (times)')
    plt.ylabel('reward')
    plt.title('reward%d' % order)
    if order == 000:
        plt.title('total reward')
        foo_fig = plt.gcf()
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        fig_name_png = file_path + '/figure_%d.png' % order
        foo_fig.savefig(fig_name_png)
    plt.legend()
    plt.grid()
    plt.show()


def main():
    topConfig = UavConfig(train_flag=2)  # 不保存数据
    # 飞行时间最短的一次路径文件
    date = r'20231130\00'
    date1 = '2023113000'
    file_path = '../rewards'
    train_flag = True

    # plot figures
    if train_flag:
        # plt.subplot(221)
        # order = 2
        # reward_file = r'C:\Users\Lenovo\PycharmProjects\uavTraj_smoothing\train\user\uav\%s\data\user%d_dis%d\train\every_reward\reward%d\uav_reward' \
        #               r'%d.npy' % (
        #                   date, topConfig.iotd_num, topConfig.total_x, order, order)
        # plot_reward(reward_file, topConfig, file_path, order=2)
        #
        # plt.subplot(222)
        # order = 4
        # reward_file = r'C:\Users\Lenovo\PycharmProjects\uavTraj_smoothing\train\user\uav\%s\data\user%d_dis%d\train\every_reward\reward%d\uav_reward' \
        #               r'%d.npy' % (
        #                   date, topConfig.iotd_num, topConfig.total_x, order, order)
        # plot_reward(reward_file, topConfig, file_path, order=4)
        #
        # plt.subplot(223)
        # order = 7
        # reward_file = r'C:\Users\Lenovo\PycharmProjects\uavTraj_smoothing\user\uav\%s\data\user%d_dis%d\train\every_reward\reward%d\uav_reward' \
        #               r'%d.npy' % (
        #                   date, topConfig.iotd_num, topConfig.total_x, order, order)
        # plot_reward(reward_file, topConfig, file_path, order=7)
        #
        # plt.subplot(224)
        # reward_file = r'C:\Users\Lenovo\PycharmProjects\uavTraj_smoothing\train\user\uav\%s\data\user%d_dis%d\train\reward\uav_reward.npy' % (
        #     date, topConfig.iotd_num, topConfig.total_x)
        # plot_reward(reward_file, topConfig, file_path, order=000)

# __________________________________________________________________________________________________________
        fig, ax1 = plt.subplots()
        file_path = 'train/'
        file = r'D:\A_codes\uavTraj_smoothing\train\user\uav\20231130\00\data\user15_dis1000\train_log_file\log_file.txt'
        f = open(file)
        line = f.readline()
        t_end = []
        rewards = []
        while line:
            if '|' in line:
                if not line[0] == '-':
                    line = line.split(' ')
                    line = [l for l in line if l.strip()]
                    if 'ep_len_mean' in line:
                        t_end.append(eval(line[3]))
                    if 'ep_rew_mean' in line:
                        rewards.append(eval(line[3]))
            line = f.readline()
        f.close()
        ax1.plot(t_end[0:40000], 'g+')
        ax1.set_ylabel('completion time', color='g',fontsize=16)
        ax2 = ax1.twinx()
        ax2.plot(rewards[0:40000], 'r')
        ax2.set_ylabel('total rewards', color='r',fontsize=16)
        plt.grid()
        ax1.set_xlabel('episodes',fontsize=16)


        plt.savefig('time.png',dpi=300)
        plt.show()
        # ---------------------------------------------------------------------------------------------------------
    # else:  # plot figures in evaluate
    #     plt.subplot(221)
    #     order = 2
    #     reward_file = r'C:\Users\Lenovo\PycharmProjects\uavTraj_smoothing\evaluate\user\uav\%s\data\user%d_dis%d\train\every_reward\reward%d' \
    #                   r'\uav_reward%d.npy' % (
    #                       date, topConfig.iotd_num, topConfig.total_x, order, order)
    #     plot_reward(reward_file, topConfig, file_path, order=order)
    #
    #     plt.subplot(222)
    #     order = 4
    #     reward_file = r'C:\Users\Lenovo\PycharmProjects\uavTraj_smoothing\evaluate\user\uav\%s\data\user%d_dis%d\train\every_reward\reward%d' \
    #                   r'\uav_reward%d.npy' % (
    #                       date, topConfig.iotd_num, topConfig.total_x, order, order)
    #     plot_reward(reward_file, topConfig, file_path, order=order)
    #
    #     plt.subplot(223)
    #     order = 7
    #     reward_file = r'C:\Users\Lenovo\PycharmProjects\uavTraj_smoothing\evaluate\user\uav\%s\data\user%d_dis%d\train\every_reward\reward%d' \
    #                   r'\uav_reward%d.npy' % (
    #                       date, topConfig.iotd_num, topConfig.total_x, order, order)
    #     plot_reward(reward_file, topConfig, file_path, order=order)
    #
    #     plt.subplot(224)
    #     reward_file = r'C:\Users\Lenovo\PycharmProjects\uavTraj_smoothing\evaluate\user\uav\%s\data\user%d_dis%d\train\reward\uav_reward.npy' % (
    #         date, topConfig.iotd_num, topConfig.total_x)
    #     plot_reward(reward_file, topConfig, file_path, order=000)


if __name__ == "__main__":
    main()
