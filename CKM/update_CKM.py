import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from CKM.err import add_err

# 加载学生模型
student_model = tf.keras.models.load_model('uav_user_channel_student_model1.h5')


# 新数据加载和预处理函数
def load_and_preprocess_new_data():
    new_inputs = np.load('new_qs.npy')
    new_channel_gains = np.load('new_channel_gains.npy')
    new_environment_features = np.load('new_all_structures.npy')

    # 归一化新环境特征
    new_environment_features[:, :, :, 0] /= 1000
    new_environment_features[:, :, :, 1] /= 1000
    new_environment_features[:, :, :, 2] /= 250

    # 归一化新目标值
    gains_min = np.min(new_channel_gains)
    gains_max = np.max(new_channel_gains)
    new_channel_gains = (new_channel_gains - gains_min) / (gains_max - gains_min)

    # 添加噪声并归一化
    noise_indices = np.random.choice(len(new_inputs), size=int(0.5 * len(new_inputs)), replace=False)
    new_inputs[noise_indices] = add_err(new_inputs[noise_indices], CEP=5)
    new_inputs = new_inputs / [1000, 1000, 250, 1000, 1000, 750]

    # 创建噪声标记
    noise_labels = np.zeros((len(new_inputs), 1))
    noise_labels[noise_indices] = 1
    new_inputs = np.hstack((new_inputs, noise_labels))

    return new_inputs, new_channel_gains, new_environment_features


# 微调模型函数
def finetune_student_model(student_model, new_inputs, new_channel_gains, new_environment_features):
    X_train, X_val, y_train, y_val, env_train, env_val = train_test_split(
        new_inputs, new_channel_gains, new_environment_features, test_size=0.2, random_state=42)

    optimizer = Adam(learning_rate=0.0001)  # 使用较小的学习率进行微调
    student_model.compile(optimizer=optimizer, loss='mse')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    tf_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1, write_images=True)

    history = student_model.fit(
        [X_train, env_train], y_train,
        epochs=15,
        batch_size=32,
        validation_data=([X_val, env_val], y_val),
        callbacks=[early_stopping, lr_scheduler, tf_callback]
    )

    return student_model, history


# 加载新数据
new_inputs, new_channel_gains, new_environment_features = load_and_preprocess_new_data()

# 微调学生模型
student_model, history = finetune_student_model(student_model, new_inputs, new_channel_gains, new_environment_features)

# 保存微调后的学生模型
student_model.save('uav_user_channel_student_model_finetuned.h5')

# 评估模型
test_loss = student_model.evaluate([new_inputs, new_environment_features], new_channel_gains)
print(f'Test loss after fine-tuning: {test_loss}')

# 可视化训练过程
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# 计算预测准确率
def calculate_accuracy(y_true, y_pred, tolerance):
    relative_error = np.abs((y_true - y_pred) / y_true)
    accuracy = np.mean(relative_error < tolerance)
    return accuracy


# 反归一化预测值和真实值
gains_min, gains_max = np.load('global_norm.npy')
y_pred = student_model.predict([new_inputs, new_environment_features]) * (gains_max - gains_min) + gains_min
new_channel_gains = new_channel_gains * (gains_max - gains_min) + gains_min

accuracy = calculate_accuracy(new_channel_gains, y_pred)
print(f'Prediction accuracy within 10% tolerance after fine-tuning: {accuracy * 100:.2f}%')
