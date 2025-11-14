import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from CKM.err import add_err
from CKM.mixed_CKM import calculate_accuracy, predict_with_noise, load_and_scale_data

# 载入全局归一化参数
gains_min, gains_max = np.load(r'D:\A_codes\uavTraj_smoothing\CKM\global_norm.npy')

# 载入模型
model = tf.keras.models.load_model(r'D:\A_codes\Trans2024\CKM\uav_user_channel_model2_1.h5')

# 载入IoT设备位置
iotd_pos = np.zeros((15, 3))
with open('UE_q.txt') as f:
    for i in range(15):
        l = f.readline().strip('\n').split(' ')
        iotd_pos[i] = ([eval(l[0]), eval(l[1]), eval(l[2])])

# 生成输入数据
# first_two = np.random.uniform(0, 1000, size=(150, 2))
# first_3 = np.random.uniform(0, 250, size=(150, 1))
# last_two_columns = np.random.uniform(0, 1000, size=(150, 2))
# last_column = np.random.uniform(250, 750, size=(150, 1))
# uav_pos = np.tile(np.array([0, 0, 250]), (150, 1))
# iotd_pos = np.tile(iotd_pos, (10, 1))
# new_inputs = np.hstack((iotd_pos, last_two_columns, last_column))
new_inputs, y, env = load_and_scale_data()
new_inputs = new_inputs[0:100,:]
y = y[0:100,:]
env = env[0:100,:]
# 加载环境特征
environment_scaled = np.load('environment_scaled.npy')
gains_min, gains_max = np.load(r'D:\A_codes\uavTraj_smoothing\CKM\global_norm.npy')
y *= (gains_max - gains_min) + gains_min  # 将预测值反归一化

# 处理数据并进行预测
for i in range(2):
    # new_inputs = np.hstack((iotd_pos, last_two_columns, last_column))
    if i == 0:
        new_inputs_with_noise = add_err(new_inputs, CEP=5)
    else:
        new_inputs_with_noise = add_err(new_inputs, CEP=25)

    # new_inputs_with_noise /= [1000, 1000, 250, 1000, 1000, 750]
    # new_inputs /= [1000, 1000, 250, 1000, 1000, 750]
    new_env = environment_scaled[:len(new_inputs)]

    pred_with_noise = predict_with_noise(model, new_inputs_with_noise, new_env, is_noisy_flag=True)
    pred_without_noise = predict_with_noise(model, new_inputs, new_env, is_noisy_flag=False)

    plt.subplot(211 + i)
    plt.plot(pred_with_noise, 'r-*', label='prediction with noise')
    plt.plot(pred_without_noise, 'g-o', label='prediction without noise')
    plt.plot(y,'b-.', label='true')
    plt.legend(fontsize=20,loc='lower left')
    if i == 0:
        plt.title(
        f'(a), Prediction accuracy within 5% tolerance: {calculate_accuracy(pred_without_noise, pred_with_noise, 0.05) * 100:.2f}%',fontsize=20)
    else:
        plt.title(
            f'(b), Prediction accuracy within 5% tolerance: {calculate_accuracy(pred_without_noise, pred_with_noise, 0.05) * 100:.2f}%',fontsize=20)

# 调整图形大小以填充屏幕并保存为全屏图片
fig = plt.gcf()
fig.set_size_inches(17.5, 10.5)  # 根据需要调整尺寸以适应屏幕
# plt.savefig('prediction.png', bbox_inches='tight', pad_inches=1, dpi=800)
plt.show()

# 打印准确性
print(
    f'Prediction accuracy within 5% tolerance: {calculate_accuracy(pred_without_noise, pred_with_noise, 0.05) * 100:.2f}%')
