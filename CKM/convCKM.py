import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate, Flatten, Add, Conv2D, GlobalAveragePooling2D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,Adadelta,Adamax
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from CKM.err import add_err
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from tensorflow.keras.layers import Attention

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# 定义注意力层
def attention_layer(inputs):
    query, value = inputs, inputs
    attention = Attention()([query, value])
    return attention

# 加载数据
def load_and_scale_data():
    inputs = np.load('qs.npy')  # 原始输入特征
    true_channel_gains = np.load('channel_gains_adjust.npy')  # 目标值
    environment_features = np.load('all_structures.npy')  # 环境特征，形状为 (num_buildings, num_faces, num_vertices_per_face, 3)
    environment_features = environment_features.astype(np.float32)

    # 标准化 environment_features
    # mean_env = np.mean(environment_features, axis=(0, 1, 2), keepdims=True)
    # std_env = np.std(environment_features, axis=(0, 1, 2), keepdims=True)
    # environment_features = (environment_features - mean_env) / std_env
    # 对 environment_features 进行归一化
    environment_features[:, :, :, 0] /= 1000  # 归一化 x 坐标
    environment_features[:, :, :, 1] /= 1000  # 归一化 y 坐标
    environment_features[:, :, :, 2] /= 250  # 归一化 z 坐标
    global gains_min
    gains_min = np.min(true_channel_gains)
    global gains_max
    gains_max = np.max(true_channel_gains)
    np.save('global_norm', [gains_min, gains_max])
    true_channel_gains = (true_channel_gains - gains_min) / (gains_max - gains_min)  # 归一化
    # 标准化 true_channel_gains
    # mean_gains = np.mean(true_channel_gains)
    # std_gains = np.std(true_channel_gains)
    # true_channel_gains = (true_channel_gains - mean_gains) / std_gains

    # 保存归一化参数
    # np.savez('global_norm.npz', mean_gains=mean_gains, std_gains=std_gains, mean_env=mean_env, std_env=std_env)

    # 获取样本数量
    num_samples = inputs.shape[0]

    # 复制环境特征到与输入样本数量相同
    environment_scaled = np.tile(environment_features, (num_samples, 1, 1, 1, 1))
    np.save('environment_scaled', environment_scaled)

    return inputs, true_channel_gains, environment_scaled


# 构建环境特征编码器
def build_environment_encoder(environment_shape):
    input_env = Input(shape=environment_shape)
    # 将输入展平为符合 Conv2D 输入要求的维度
    reshaped_env = Reshape((environment_shape[0] * environment_shape[1], environment_shape[2], environment_shape[3]))(input_env)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(reshaped_env)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    model = Model(inputs=input_env, outputs=x)
    return model

# 构建模型
def build_model(input_dim, environment_shape):
    # 输入层
    input_layer = Input(shape=(input_dim + 1,))  # 输入维度增加1，用于噪声标记
    environment_input = Input(shape=environment_shape)

    # 环境特征编码
    environment_encoder = build_environment_encoder(environment_shape)
    encoded_environment = environment_encoder(environment_input)

    # 添加注意力层
    attended_environment = attention_layer(encoded_environment)

    # 将输入特征和环境特征连接
    concatenated = Concatenate()([input_layer, attended_environment])
    reshaped = Reshape((concatenated.shape[1], 1))(concatenated)  # 调整形状以适配卷积层

    # # 使用一维卷积层替换原有MLP部分
    x = Conv1D(64, kernel_size=1, activation='relu', padding='same')(reshaped)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
    x = GlobalAveragePooling1D()(x)
    # x = Conv1D(64, kernel_size=5, activation='relu', padding='same')(reshaped)
    # x = BatchNormalization()(x)
    # x = MaxPooling1D(pool_size=2)(x)
    # x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    # x = BatchNormalization()(x)
    # x = MaxPooling1D(pool_size=2)(x)
    # x = Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)  # 增加Dropout以提高泛化能力
    # x = GlobalAveragePooling1D()(x)

    # 输出层
    output_layer = Dense(1)(x)

    # 创建模型
    model = Model(inputs=[input_layer, environment_input], outputs=output_layer)
    return model

def calculate_accuracy(y_true, y_pred, tolerance=0.1):
    relative_error = np.abs((y_true - y_pred) / y_true)
    accuracy = np.mean(relative_error < tolerance)
    return accuracy

def predict_with_noise(model, inputs, environment, is_noisy_flag):
    # norm_params = np.load('../CKM/global_norm.npz')
    # mean_gains = norm_params['mean_gains']
    # std_gains = norm_params['std_gains']
    gains_min,gains_max = np.load('global_norm.npy')
    noise_labels = np.ones((inputs.shape[0], 1)) if is_noisy_flag else np.zeros((inputs.shape[0], 1))
    inputs_with_noise_flag = np.hstack((inputs, noise_labels))
    predictions = model.predict([inputs_with_noise_flag, environment], verbose=0)
    # predictions = predictions * (gains_max-gains_min) + gains_min
    # predictions = predictions * (gains_max-gains_min) + gains_min
    # predictions = (predictions - 0.4) / (0.5 - 0.4) * (6 - (-6)) + (-6)
    # predictions = 1 / (1 + np.exp(-predictions))
    return predictions

def main():
    # 确保每次训练时数据是随机的
    tf.random.set_seed(42)
    gains_min, gains_max = np.load('global_norm.npy')
    inputs, gains_scaled, environment_scaled = load_and_scale_data()
    # norm_params = np.load('global_norm.npz')
    # mean_gains = norm_params['mean_gains']
    # std_gains = norm_params['std_gains']


    # 添加所有噪声
    inputs_with_n = add_err(inputs, CEP=5)
    # inputs_with_n = (inputs_with_n - np.mean(inputs_with_n, axis=0)) / np.std(inputs_with_n, axis=0)
    # inputs = (inputs - np.mean(inputs, axis=0)) / np.std(inputs, axis=0)
    inputs_with_n /= [1000, 1000, 250, 1000, 1000, 750]
    inputs /= [1000, 1000, 250, 1000, 1000, 750]
    labels1 = np.ones((len(inputs),1))
    labels2 = np.zeros((len(inputs),1))
    inputs = np.vstack((inputs_with_n,inputs))
    labels = np.vstack((labels1,labels2))

    # 将噪声标记添加到输入特征中
    inputs_scaled_with_noise = np.hstack((inputs, labels))
    gains_scaled = np.vstack((gains_scaled,gains_scaled))
    environment_scaled = np.tile(environment_scaled, (2, 1, 1, 1, 1))
    # 检查数据形状是否一致


    X_train, X_test, y_train, y_test, env_train, env_test = train_test_split(
        inputs_scaled_with_noise, gains_scaled, environment_scaled, test_size=0.3, random_state=42)
    model = build_model(input_dim=X_train.shape[1] - 1, environment_shape=env_train.shape[1:])
    # optimizer = Adam(learning_rate=0.0001)

    learning_rate = 0.0001  # 尝试不同的学习率
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='mse')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau (monitor='val_loss', factor=0.5, patience=5)
    tf_callback = tf.keras.callbacks.TensorBoard(log_dir="../logs1", histogram_freq=1, write_images=True)

    history = model.fit(
        [X_train, env_train], y_train,
        epochs=7,  # 增加训练轮数
        batch_size=32,  # 使用较小的批处理大小
        validation_split=0.3,
        callbacks=[early_stopping, lr_scheduler, tf_callback]
    )

    model.save('uav_user_channel_model_conv.h5')

    train_loss = model.evaluate([X_train, env_train], y_train,batch_size=32)
    print(f'Train loss: {train_loss}')

    test_loss = model.evaluate([X_test, env_test], y_test)
    print(f'Test loss: {test_loss}')

    y_pred = model.predict([X_test, env_test])
    y_pred_rescaled = y_pred * (gains_max-gains_min) + gains_min  # 进行去标准化
    y_test_rescaled = y_test * (gains_max-gains_min) + gains_min  # 对真实标签去标准化
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


    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    accuracy = calculate_accuracy(y_test, y_pred,0.1)
    print(f'Prediction accuracy within 10% tolerance: {accuracy * 100:.2f}%')

    iotd_pos = np.zeros((15, 3))
    with open('UE_q.txt') as f:
        for i in range(15):
            l = f.readline().strip('\n').split(' ')
            iotd_pos[i] = ([eval(l[0]), eval(l[1]), eval(l[2])])
    iotd_pos = np.tile(iotd_pos, (100, 1))
    last_two_columns = np.random.uniform(0, 1000, size=(1500, 2))
    last_column = np.random.uniform(250, 750, size=(1500, 1))
    # 合并所有列形成150x6的数组
    new_inputs = np.hstack((iotd_pos, last_two_columns, last_column))
    environment_scaled = np.load('environment_scaled.npy')
    new_inputs_with_noise = add_err(new_inputs, CEP=5)  # 示例加了噪声的新输入数据
    new_inputs_with_noise /= [1000, 1000, 250, 1000, 1000, 750]
    new_inputs /= [1000, 1000, 250, 1000, 1000, 750]
    new_env = environment_scaled[0:len(new_inputs)]  # 示例新的环境特征
    pred_with_noise = predict_with_noise(model, new_inputs_with_noise,new_env,is_noisy_flag=True)
    pred_without_noise = predict_with_noise(model, new_inputs,new_env, is_noisy_flag=False)

    plt.figure()
    plt.plot(pred_without_noise, 'g-o', label='pred_without_noise')
    plt.plot(pred_with_noise, 'r*', label='pred_with_noise')
    plt.legend()
    plt.savefig('show_rand.jpg', dpi=300)
    plt.show()

    print(
        f'Prediction accuracy within 5% tolerance: {calculate_accuracy(pred_without_noise, pred_with_noise, 0.05) * 100:.2f}%')
    print(
        f'Prediction accuracy within 1% tolerance: {calculate_accuracy(pred_without_noise, pred_with_noise, 0.01) * 100:.2f}%')

if __name__ == '__main__':
    main()
