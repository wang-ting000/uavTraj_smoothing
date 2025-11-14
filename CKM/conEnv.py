import pickle
import numpy as np
import glob
import os

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def extract_coordinates(file):
    with open(file, 'r') as f:
        in_sub_structure = False
        in_face = False
        collect_vertices = False
        file_data = []
        sub_structure_data = []
        vertices_data = []

        for line in f:
            line = line.strip()

            if 'begin_<sub_structure>' in line:
                in_sub_structure = True
                sub_structure_data = []  # 重置子结构数据
            elif 'end_<sub_structure>' in line:
                in_sub_structure = False
                if sub_structure_data:
                    file_data.append(sub_structure_data)  # 保存子结构数据

            if in_sub_structure:
                if 'begin_<face>' in line:
                    in_face = True
                    collect_vertices = False
                    vertices_data = []  # 重置顶点数据
                elif 'end_<face>' in line and in_face:
                    in_face = False
                    collect_vertices = False
                    if vertices_data:
                        sub_structure_data.append(vertices_data)  # 保存顶点数据
                elif in_face:
                    if 'nVertices' in line:
                        collect_vertices = True  # 开始收集顶点数据
                    elif collect_vertices and line:
                        vertices = list(map(float, line.split()))
                        vertices_data.append(vertices)  # 将顶点添加到列表中

        return file_data if file_data else None


def save_with_pickle(data, filename):
    """
    使用pickle将数据保存到文件
    """
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def format_structure(sub_structure):
    bottom_face = sub_structure[0]
    top_face = sub_structure[-1]
    sides = sub_structure[1:-1]

    # 将每个面的点的坐标转换为单独的 numpy 数组
    formatted_structure = [
        [np.array(point) for point in bottom_face],  # 底面
        [np.array(point) for point in top_face],  # 顶面
        *[[np.array(point) for point in side] for side in sides]  # 侧面
    ]
    return formatted_structure


def main():
    save_all_struct = []
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 初始化一个列表来保存所有地面的 z 轴值
    ground_z_values = []

    def process_files(pattern, edgecolor, facecolor, alpha):
        path = 'D:/wireless insite/save_objects'
        path_pattern = os.path.join(path, pattern)
        files = glob.glob(path_pattern, recursive=True)

        all_structures = []  # 存放所有文件的对象坐标

        for file in files:
            coords = extract_coordinates(file)
            if coords:
                for sub_structure_data in coords:
                    formatted_structure = format_structure(sub_structure_data)
                    if formatted_structure:
                        all_structures.append(formatted_structure)

                        # 记录地面的 z 轴值
                        ground_z_values.extend([vertex[2] for vertex in formatted_structure[0]])
        save_all_struct.append(all_structures)
        save_with_pickle(all_structures, f'all_structures_{pattern.split(".")[-1]}.pkl')

        for structure in all_structures:
            poly3d = Poly3DCollection(structure, alpha=alpha, linewidths=0.5, edgecolors=edgecolor)
            poly3d.set_facecolor(facecolor)
            ax.add_collection3d(poly3d)
        return save_all_struct
    # process_files('**/*.ter', 'brown', [0.55, 0.27, 0.07, 1], 0.25)
    process_files('**/*.flp', 'k', [0.5, 0.5, 0.5, 0.9], 0.25)
    process_files('**/*.veg', 'green', [0, 0.39, 0, 0.1], 0.25)
    process_files('**/*.city', 'black', [0.66, 0.66, 0.66, 0.5], 0.25)
    process_files('**/*.object', 'blue', [0.68, 0.85, 0.9, 0.5], 0.25)

    data = np.loadtxt('UE_q.txt')
    # 提取坐标
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    # 绘制散点图，使用颜色映射（colormap）和更大的标记
    sc = ax.scatter(x, y, z, c=z, cmap='viridis', marker='o', s=50, alpha=0.7)

    # 添加颜色条
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Z Value')
    ax.grid(True)

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 设置显示范围
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_zlim(0, 1000)

    plt.show()


if __name__ == "__main__":
    main()
    all_dat = []
    for i in ['city','object']:
        with open('all_structures_%s.pkl'%(i),'rb') as f:
            dat = pickle.load(f)
            # for i in dat:
            #     print(np.shape(i))
            # print('---')
            all_dat += dat
    np.save('all_struct.npy',np.array(all_dat,dtype=object),allow_pickle=True)



