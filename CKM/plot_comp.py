import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# 读取PNG文件
top_view_png = 'CEP对比_俯视图.png'
side_view_png = 'CEP对比_侧视图.png'

# 打开PNG图像
top_view_image = Image.open(top_view_png)
side_view_image = Image.open(side_view_png)

# 将俯视图缩小
top_view_image_resized = top_view_image.resize((100, 80))  # 调整缩小尺寸

# 将俯视图转换为数组以便在Matplotlib中显示
top_view_array = np.array(top_view_image_resized)
side_view_array = np.array(side_view_image)

# 创建绘图
fig, ax = plt.subplots(figsize=(20, 16))

# 显示侧视图
ax.imshow(side_view_array)
ax.axis('off')

# 在侧视图上添加俯视图
x_offset = 300  # 调整位置
y_offset = 300  # 调整位置
ax.imshow(top_view_array, extent=(x_offset, x_offset + top_view_array.shape[1], y_offset, y_offset + top_view_array.shape[0]), aspect='auto')

# 添加箭头
arrow_start = (x_offset + top_view_array.shape[1] / 2, y_offset + top_view_array.shape[0])
arrow_end = (side_view_array.shape[1] / 2, side_view_array.shape[0] / 2)
ax.annotate('', xy=arrow_end, xytext=arrow_start, arrowprops=dict(facecolor='red', arrowstyle='->'))

# 添加标题
plt.title('Influences of CEP for UAV Trajectory Design')

# 保存和显示图像
output_path = 'combined_image_with_arrow.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()
