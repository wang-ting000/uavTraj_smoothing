import numpy as np
import matplotlib.pyplot as plt


def compare_fading():
    # 读取两个npy文件
    channel_gains = np.load('channel_gains.npy')
    channel_gains_new = np.load('channel_gains_new1.npy')

    # 确保两个数组的形状一致
    if channel_gains.shape != channel_gains_new.shape:
        print("两个文件的形状不一致，无法比较。")
        return

    # 创建索引数组
    indices = np.arange(len(channel_gains))

    # 绘制对比图
    plt.figure(figsize=(10, 6))
    plt.plot(indices, channel_gains, label='Original Channel Gains', color='blue', alpha=0.6)
    plt.plot(indices, channel_gains_new, label='New Channel Gains', color='red', alpha=0.6)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Comparison of Channel Gains')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 计算数值差距
    differences = channel_gains - channel_gains_new

    # 绘制直方图并获取频率和横坐标
    plt.figure(figsize=(10, 6))
    counts, bins, _ = plt.hist(differences, bins=50, color='green', alpha=0.7)
    plt.xlabel('Difference Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Channel Gains Differences')
    plt.grid(True)
    plt.show()

    # 打印总样本数
    total_samples = len(differences)
    print(f"总样本数: {total_samples}")

    # 找出频率最大的三个数及其对应的横坐标
    top_three_indices = np.argsort(counts)[-3:][::-1]  # 降序排列
    print("直方图中频率最大的三个数及其对应的横坐标:")
    for i, index in enumerate(top_three_indices, 1):
        bin_center = (bins[index] + bins[index + 1]) / 2  # 计算bin的中心点作为横坐标
        print(f"第{i}大频率: 频率 {counts[index]}, 横坐标 {bin_center}")


if __name__ == "__main__":
    compare_fading()