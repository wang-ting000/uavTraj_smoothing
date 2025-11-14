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


def format_structure(sub_structure):
    if len(sub_structure) < 2:
        print(f"Not enough faces to format a structure: {sub_structure}")
        return None  # 如果面数量不足，返回None

    bottom_face = sub_structure[0]
    top_face = sub_structure[-1]

    sides = sub_structure[1:-1]

    if len(bottom_face) < 4 or len(top_face) < 4 or len(sides) < 4:
        print(
            f"Not enough vertices or sides to format a structure: bottom_face {bottom_face}, top_face {top_face}, sides {sides}")
        return None  # 如果顶点数量不足，返回None

    # 将每个面的点的坐标转换为单独的 numpy 数组
    formatted_structure = [
        [np.array(point) for point in bottom_face],  # 底面
        [np.array(point) for point in top_face],  # 顶面
        *[[np.array(point) for point in side] for side in sides]  # 侧面
    ]
    return formatted_structure


def main():
    path = 'D:/wireless insite/save_objects'
    pattern = '**/*.veg'
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

    # 打印或处理 all_structures
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for structure in all_structures:
        print(structure)
        poly3d = Poly3DCollection(structure, alpha=.25, linewidths=1, edgecolors='r')
        poly3d.set_facecolor([0.5, 0.5, 1, 0.1])  # 设置颜色（蓝色，带透明度）

        # 将多边形添加到图形中
        ax.add_collection3d(poly3d)

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
