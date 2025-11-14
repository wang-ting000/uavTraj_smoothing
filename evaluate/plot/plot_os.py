import glob
import numpy as np
from CKM.conEnv import extract_coordinates, format_structure
from uav_gym.envs.uav_env import UavConfig
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def myplot(pos,pos1, top_config, t, iotd_pos, payload, save_path='updates.png'):
    fig = plt.figure(1, figsize=(18, 9))
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()  # toggle fullscreen mode

    def plot_environment(ax):
        process_files(ax, '**/*.flp', 'k', [0.5, 0.5, 0.5, 0.9], 0.25)
        process_files(ax, '**/*.veg', 'green', [0, 0.39, 0, 0.1], 0.25)
        process_files(ax, '**/*.city', 'black', [0.66, 0.66, 0.66, 0.5], 0.25)
        process_files(ax, '**/*.object', 'blue', [0.68, 0.85, 0.9, 0.5], 0.25)

    ax1 = fig.add_subplot(121, projection="3d")
    plot_environment(ax1)
    ax1.view_init(10, 0)  # 调整视角
    ax1.set_xlabel('X(m)')
    ax1.set_ylabel('Y(m)')
    ax1.set_zlabel('Z(m)')
    ax1.scatter(top_config.UAV_INITIAL_POSITION_X, top_config.UAV_INITIAL_POSITION_Y,
                top_config.UAV_INITIAL_POSITION_Z, c='r', marker='x', s=30)

    ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], color='#4B0082', marker='^', label='UAV trajectory of OS-PPO, %ds'%(len(pos)))
    ax1.plot(pos1[:, 0], pos1[:, 1], pos1[:, 2], color='m', marker='*', label='UAV trajectory of PEC-PPO, %ds'%(len(pos1)))
    track1 = np.load(r'D:\A_codes\uavTraj_smoothing\CKM\track3.npy')

    # ax1.plot(track1[:, 0], track1[:, 1], track1[:, 2], color='#4b0082', marker='+', label='UAV trajectory of CKM-PPO')

    X = np.asarray([iotd_pos[i][0] for i in range(top_config.iotd_num)])
    Y = np.asarray([iotd_pos[i][1] for i in range(top_config.iotd_num)])
    Z = np.asarray([iotd_pos[i][2] for i in range(top_config.iotd_num)])
    p1 = ax1.scatter(X, Y, Z, edgecolors='k', linewidths=1, c=payload[-1], vmin=0, vmax=1, cmap='viridis', alpha=0.75,
                     s=50, marker='o', label='IoTDs')

    los_pos = np.load(r'D:\A_codes\uavTraj_smoothing\evaluate\plot\pos.npy')
    # ax1.plot(los_pos[:, 0], los_pos[:, 1], los_pos[:, 2], color='#8b0000', marker='*',
    #          label='UAV trajectory of LoS-PPO')

    cbar1 = plt.colorbar(p1, ax=ax1, pad=-0.1, shrink=0.5,aspect=30)  # 调整色条大小
    cbar1.set_label('Payload',fontsize=20)

    ax1.legend(loc='upper right',fontsize=16)

    ax2 = fig.add_subplot(122, projection="3d")
    plot_environment(ax2)
    ax2.view_init(90, 0)
    ax2.set_xlabel('X(m)')
    ax2.set_ylabel('Y(m)')
    ax2.set_zlabel('Z(m)')

    ax2.plot(pos[:, 0], pos[:, 1], pos[:, 2], color='#4B0082', marker='^', label='UAV trajectory of OS-PPO, %ds'%(len(pos)))
    ax2.plot(pos1[:, 0], pos1[:, 1], pos1[:, 2], color='m', marker='*', label='UAV trajectory of PEC-PPO, %ds'%(len(pos1)))

    # ax2.plot(track1[:, 0], track1[:, 1], track1[:, 2], color='#4b0082', marker='+', label='UAV trajectory of CKM-PPO')

    p2 = ax2.scatter(X, Y, Z, edgecolors='k', linewidths=1, c=payload[-1], vmin=0, vmax=1, cmap='viridis', alpha=0.75,
                     s=50, marker='o', label='IoTDs')

    # ax2.plot(los_pos[:, 0], los_pos[:, 1], los_pos[:, 2], color='#8b0000', marker='*',
    #          label='UAV trajectory of LoS-PPO')

    cbar2 = plt.colorbar(p2, ax=ax2, pad=-0.1, shrink=0.5,aspect=30)  # 调整色条大小
    cbar2.set_label('Payload',fontsize=20)

    ax2.legend(loc='upper right',fontsize=16)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=-0.05)  # 调整子图间距和边距

    # 保存全屏图片
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def process_files(ax, pattern, edgecolor, facecolor, alpha):
    path = 'D:/wireless insite/save_objects'
    path_pattern = os.path.join(path, pattern)
    files = glob.glob(path_pattern, recursive=True)

    all_structures = []

    for file in files:
        coords = extract_coordinates(file)
        if coords:
            for sub_structure_data in coords:
                formatted_structure = format_structure(sub_structure_data)
                if formatted_structure:
                    all_structures.append(formatted_structure)

    for structure in all_structures:
        poly3d = Poly3DCollection(structure, alpha=alpha, linewidths=0.5, edgecolors=edgecolor)
        poly3d.set_facecolor(facecolor)
        ax.add_collection3d(poly3d)


def main():
    topConfig = UavConfig(train_flag=2)  # 不保存数据
    with open(r'D:\A_codes\uavTraj_smoothing\train\output_os.txt', 'r') as f:
        lines = f.readlines()
        pos = np.zeros((len(lines) // 3, 3))
        for i in range(len(lines) // 3):
            pos[i, 0] = eval(lines[i * 3 + 0]) * 1000
            pos[i, 1] = eval(lines[i * 3 + 1]) * 1000
            pos[i, 2] = eval(lines[i * 3 + 2]) * 750
    with open(r'D:\A_codes\uavTraj_smoothing\train\output.txt', 'r') as f:
        lines = f.readlines()
        pos1 = np.zeros((len(lines) // 3, 3))
        for i in range(len(lines) // 3):
            pos1[i, 0] = eval(lines[i * 3 + 0]) * 1000
            pos1[i, 1] = eval(lines[i * 3 + 1]) * 1000
            pos1[i, 2] = eval(lines[i * 3 + 2]) * 750
        pos1[-1][-1] = 250
    payload = np.load('payload.npy')
    t = len(payload) + 1
    iotd_pos = np.load('iotd_pos.npy')

    myplot(pos,pos1, topConfig, t, iotd_pos, payload)


if __name__ == '__main__':
    main()
