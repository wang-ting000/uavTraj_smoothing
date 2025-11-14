import os
import numpy as np
import tensorflow as tf
from keras.layers import Lambda
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate, Flatten, Add, Conv2D, GlobalAveragePooling2D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from CKM.err import add_err

# 加载数据
def load_and_scale_data():
    inputs = np.load('qs.npy')  # 原始输入特征
    true_channel_gains = np.load(r'D:\A_codes\Trans2024\generated_data\channel_gains.npy')  # 目标值
    environment_features = np.load('all_structures.npy')  # 环境特征，形状为 (num_buildings, num_faces, num_vertices_per_face, 3)
    environment_features = environment_features.astype(np.float32)
    # 对 environment_features 进行归一化
    environment_features[:, :, :, 0] /= 1000  # 归一化 x 坐标
    environment_features[:, :, :, 1] /= 1000  # 归一化 y 坐标
    environment_features[:, :, :, 2] /= 250  # 归一化 z 坐标
    global gains_min
    gains_min = np.min(true_channel_gains)
    global gains_max
    gains_max = np.max(true_channel_gains)
    np.save('global_norm',[gains_min,gains_max])
    true_channel_gains = (true_channel_gains - gains_min) / (gains_max - gains_min)  # 归一化

    # 获取样本数量
    num_samples = inputs.shape[0]

    # 复制环境特征到与输入样本数量相同
    environment_scaled = np.tile(environment_features, (num_samples, 1, 1, 1, 1))
    np.save('environment_scaled', environment_scaled)

    return inputs, true_channel_gains, environment_scaled

# 构建环境特征编码器
# def build_environment_encoder(environment_shape):
#     input_env = Input(shape=environment_shape)
#     # 将输入展平为符合 Conv2D 输入要求的维度
#     reshaped_env = Reshape((environment_shape[0] * environment_shape[1], environment_shape[2], environment_shape[3]))(input_env)
#     x = Conv2D(32, (3, 3), activation='relu', padding='same')(reshaped_env)
#     x = BatchNormalization()(x)
#     x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
#     x = BatchNormalization()(x)
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(256, activation='relu')(x)
#     x = Dense(128, activation='relu')(x)
#     model = Model(inputs=input_env, outputs=x)
#     return model
def build_environment_encoder(environment_shape):
    input_env = Input(shape=environment_shape)
    # 将输入展平为一维
    flattened_env = Flatten()(input_env)
    x = Dense(128, activation='relu')(flattened_env)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation='relu')(x)
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

    # 将输入特征和环境特征连接
    concatenated = Concatenate()([input_layer, encoded_environment])

    # 添加残差连接的隐藏层
    def res_block(x, units):
        shortcut = Dense(units)(x)  # 调整 shortcut 的维度以匹配残差块的输出
        x = Dense(units, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dense(units, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Add()([shortcut, x])
        return x

    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(concatenated)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = res_block(x, 256)
    x = Dropout(0.1)(x)
    x = res_block(x, 128)
    x = Dropout(0.1)(x)
    x = res_block(x, 64)
    x = Dropout(0.1)(x)
    x = res_block(x, 32)
    x = Dropout(0.1)(x)

    # 输出层
    output_layer = Dense(1, activation='sigmoid')(x)


    model = Model(inputs=[input_layer, environment_input], outputs=output_layer)
    return model

def predict_with_noise(model, inputs, environment, is_noisy_flag):
    gains_min, gains_max = np.load('global_norm.npy')
    noise_labels = np.ones((inputs.shape[0], 1)) if is_noisy_flag else np.zeros((inputs.shape[0], 1))
    inputs_with_noise_flag = np.hstack((inputs, noise_labels))
    if len(environment) == 1:
        predictions = model.predict([inputs_with_noise_flag], verbose=0)
    else:
        predictions = model.predict([inputs_with_noise_flag, environment], verbose=0)
    predictions = predictions * (gains_max - gains_min) + gains_min  # 将预测值反归一化


    return predictions
def calculate_accuracy(y_true, y_pred, tolerance):
    relative_error = np.abs((y_true - y_pred) / y_true)
    accuracy = np.mean(relative_error < tolerance)
    return accuracy
# 主函数
def main():
    # 确保每次训练时数据是随机的
    np.random.seed(None)
    tf.random.set_seed(None)
    inputs, gains_scaled, environment_scaled = load_and_scale_data()

    # 添加部分数据噪声
    noise_indices = np.random.choice(len(inputs), size=int(0.5 * len(inputs)), replace=False)
    inputs_scaled_with_noise = inputs.copy()
    inputs_scaled_with_noise[noise_indices] = add_err(inputs_scaled_with_noise[noise_indices], CEP=5)
    inputs_scaled_with_noise = inputs_scaled_with_noise / [1000, 1000, 250, 1000, 1000, 750] # 归一化
    # 创建噪声标记
    noise_labels = np.zeros((len(inputs), 1))
    noise_labels[noise_indices] = 1
    # 将噪声标记添加到输入特征中
    inputs_scaled_with_noise = np.hstack((inputs_scaled_with_noise, noise_labels))

    # 检查数据形状是否一致
    num_samples = inputs_scaled_with_noise.shape[0]
    assert gains_scaled.shape[0] == num_samples, "Number of samples in gains_scaled does not match inputs_scaled"
    assert environment_scaled.shape[0] == num_samples, "Number of samples in environment_scaled does not match inputs_scaled"

    X_train, X_test, y_train, y_test, env_train, env_test = train_test_split(
        inputs_scaled_with_noise, gains_scaled, environment_scaled, test_size=0.2, random_state=42,shuffle=True)

    model = build_model(input_dim=X_train.shape[1] - 1, environment_shape=env_train.shape[1:])
    optimizer = Adam(learning_rate=0.0001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    tf_callback = tf.keras.callbacks.TensorBoard(log_dir="../logs", histogram_freq=1, write_images=True)

    history = model.fit(
        [X_train, env_train], y_train,
        epochs=300,
        batch_size=32,  # 使用较小的批处理大小
        validation_split=0.3,
        callbacks=[early_stopping, lr_scheduler, tf_callback]
    )

    model.save('uav_user_channel_model2_1.h5')



    y_pred = model.predict([X_test, env_test])
    # y_pred[y_pred > -0.4] *= 0.8
    # y_pred = y_pred * (gains_max - gains_min) + gains_min
    # y_test = y_test * (gains_max - gains_min) + gains_min
    plt.figure()
    plt.plot(y_pred[0:30], 'r', label='Predicted')
    plt.plot(y_test[0:30], 'g', label='True')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(y_test, 'g', label='True')
    plt.plot(y_pred, 'r', label='Predicted')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 计算预测准确率


    accuracy = calculate_accuracy(y_test, y_pred,0.1)
    print(f'Prediction accuracy within 10% tolerance: {accuracy * 100:.2f}%')

    # 对新数据进行预测，可以选择是否已经添加噪声

    # 示例预测
    first_two_columns = np.random.uniform(0, 1000, size=(1500, 2))
    third_column = np.random.uniform(0, 250, size=(1500, 1))

    last_two_columns = np.random.uniform(0, 1000, size=(1500, 2))
    last_column = np.random.uniform(250, 750, size=(1500, 1))
    # 合并所有列形成150x6的数组
    new_inputs = np.hstack((first_two_columns, third_column, last_two_columns, last_column))

    new_inputs_with_noise = add_err(new_inputs, CEP=5)  # 示例加了噪声的新输入数据
    new_inputs_with_noise = new_inputs_with_noise / [1000, 1000, 250, 1000, 1000, 750]
    new_inputs = new_inputs / [1000, 1000, 250, 1000, 1000, 750]
    new_env = environment_scaled[0:len(new_inputs)]  # 示例新的环境特征
    pred_with_noise = predict_with_noise(model, new_inputs_with_noise, new_env, is_noisy_flag=True)
    pred_without_noise = predict_with_noise(model, new_inputs, new_env, is_noisy_flag=False)

    plt.figure()
    plt.plot(pred_without_noise,'r-*',label='pred_without_noise')
    plt.plot(pred_with_noise,'g-o',label='pred_with_noise')
    plt.legend()
    plt.savefig('show1.jpg', dpi=300)
    plt.show()

    print(f'Prediction accuracy within 1% tolerance: {calculate_accuracy(pred_without_noise,pred_with_noise,0.01) * 100:.2f}%')

if __name__ == '__main__':
    main()
