import os
import numpy as np
import tensorflow as tf
from keras.activations import swish
from keras.layers import LeakyReLU, Activation, GlobalAveragePooling1D, Reshape, Conv1D, MaxPooling1D
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split

from CKM.convCKM import build_environment_encoder
from CKM.err import add_err
from CKM.convCKM import load_and_scale_data
from CKM.convCKM import predict_with_noise

# 加载教师模型
teacher_model = tf.keras.models.load_model('uav_user_channel_model_conv.h5')
def build_student_model(input_dim, environment_shape):
    # 输入层
    input_layer = Input(shape=(input_dim + 1,))
    environment_input = Input(shape=environment_shape)

    # 环境特征编码，使用简化的结构
    environment_encoder = build_environment_encoder(environment_shape)
    encoded_environment = environment_encoder(environment_input)

    # 添加简化的卷积层
    concatenated = Concatenate()([input_layer, encoded_environment])
    reshaped = Reshape((concatenated.shape[1], 1))(concatenated)
    x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(reshaped)
    x = MaxPooling1D(pool_size=3)(x)
    x = GlobalAveragePooling1D()(x)

    # 输出层
    output_layer = Dense(1)(x)

    # 创建模型
    model = Model(inputs=[input_layer, environment_input], outputs=output_layer)
    return model


# 蒸馏损失函数
def distillation_loss(temperature=3):
    def loss(y_true, y_pred):
        # 计算教师和学生输出的温度调整的MSE
        y_true_adjusted = y_true / temperature
        y_pred_adjusted = y_pred / temperature
        return tf.reduce_mean(tf.square(y_true_adjusted - y_pred_adjusted))
    return loss

# 主函数
def main():
    # 确保每次训练时数据是随机的
    np.random.seed(None)
    tf.random.set_seed(None)
    inputs_scaled, gains_scaled, environment_scaled = load_and_scale_data()

    # 添加部分数据噪声
    noise_indices = np.random.choice(len(inputs_scaled), size=int(0.5 * len(inputs_scaled)), replace=False)
    inputs_scaled_with_noise = inputs_scaled.copy()
    inputs_scaled_with_noise[noise_indices] = add_err(inputs_scaled_with_noise[noise_indices], CEP=5)

    inputs_scaled_with_noise = (inputs_scaled_with_noise - np.mean(inputs_scaled_with_noise, axis=0)) / np.std(inputs_scaled_with_noise, axis=0)
    inputs_scaled = (inputs_scaled - np.mean(inputs_scaled, axis=0)) / np.std(inputs_scaled, axis=0)

    # 创建噪声标记
    noise_labels = np.zeros((len(inputs_scaled), 1))
    noise_labels[noise_indices] = 1

    # 将噪声标记添加到输入特征中
    inputs_scaled_with_noise = np.hstack((inputs_scaled_with_noise, noise_labels))

    # 检查数据形状是否一致
    num_samples = inputs_scaled.shape[0]
    assert gains_scaled.shape[0] == num_samples, "Number of samples in gains_scaled does not match inputs_scaled"
    assert environment_scaled.shape[0] == num_samples, "Number of samples in environment_scaled does not match inputs_scaled"

    X_train, X_test, y_train, y_test, env_train, env_test = train_test_split(
        inputs_scaled_with_noise, gains_scaled, environment_scaled, test_size=0.2, random_state=42)
    # 假设 X_train, env_train 已准备好
    teacher_predictions = teacher_model.predict([X_train, env_train])
    student_model = build_student_model(input_dim=X_train.shape[1] - 1, environment_shape=env_train.shape[1:])
    optimizer = Adam(learning_rate=0.2)
    student_model.compile(optimizer=optimizer, loss='mse')
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5)
    student_model.fit([X_train, env_train], teacher_predictions, epochs=400, batch_size=32,callbacks=lr_scheduler)

    student_model.save('uav_user_channel_student_model1.h5')

    y_pred = student_model.predict([X_test, env_test])
    norm_params = np.load('global_norm.npz')
    mean_gains = norm_params['mean_gains']
    std_gains = norm_params['std_gains']
    y_pred_rescaled = y_pred * std_gains + mean_gains  # 进行去标准化
    y_test_rescaled = y_test * std_gains + mean_gains  # 对真实标签去标准化
    plt.figure()
    plt.plot(y_pred_rescaled, 'r', label='Predicted')
    plt.plot(y_test_rescaled, 'g', label='True')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(y_test_rescaled, 'g', label='True')
    plt.plot(y_pred_rescaled, 'r', label='Predicted')
    plt.legend()
    plt.show()


    # 计算预测准确率
    def calculate_accuracy(y_true, y_pred, tolerance=0.1):
        relative_error = np.abs((y_true - y_pred) / y_true)
        accuracy = np.mean(relative_error < tolerance)
        return accuracy

    accuracy = calculate_accuracy(y_test, y_pred)
    print(f'Prediction accuracy within 10% tolerance: {accuracy * 100:.2f}%')

    first_two_columns = np.random.uniform(0, 1000, size=(1500, 2))
    third_column = np.random.uniform(0, 250, size=(1500, 1))

    last_two_columns = np.random.uniform(0, 1000, size=(1500, 2))
    last_column = np.random.uniform(250, 750, size=(1500, 1))
    # 合并所有列形成150x6的数组
    new_inputs = np.hstack((first_two_columns, third_column, last_two_columns, last_column))

    new_inputs_with_noise = add_err(new_inputs, CEP=5)  # 示例加了噪声的新输入数据
    new_inputs_with_noise = (new_inputs_with_noise - np.mean(new_inputs_with_noise, axis=0)) / np.std(new_inputs_with_noise, axis=0)
    new_inputs = (new_inputs - np.mean(new_inputs, axis=0)) / np.std(new_inputs, axis=0)
    new_env = environment_scaled[0:len(new_inputs)]  # 示例新的环境特征
    pred_with_noise = predict_with_noise(student_model, new_inputs_with_noise, new_env, is_noisy_flag=True)
    pred_without_noise = predict_with_noise(student_model, new_inputs, new_env, is_noisy_flag=False)

    plt.figure()
    plt.plot(pred_without_noise,'r*',label='pred_without_noise')
    plt.plot(pred_with_noise,'g-',label='pred_with_noise')
    plt.legend()
    plt.show()
    print()
    print(f'Prediction accuracy within 10% tolerance: {calculate_accuracy(pred_without_noise, pred_with_noise) * 100:.2f}%')
if __name__ == '__main__':
    main()
