import numpy as np
import tensorflow as tf
from keras.regularizers import l2
from tensorflow.keras.layers import Lambda, Dense, Dropout, BatchNormalization, Input, Concatenate, Flatten, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Knowledge module: LoS probability channel loss model
def calculate_distance_and_theta(user_pos, uav_pos):
    # Use TensorFlow operations instead of NumPy
    horizontal_distance = tf.sqrt(tf.reduce_sum(tf.square(tf.slice(user_pos, [0], [2]) - tf.slice(uav_pos, [0], [2])))) + 1e-6  # Prevent division by 0
    real_distance = tf.sqrt(tf.square(horizontal_distance) + tf.square(user_pos[2] - uav_pos[2]))
    theta = tf.atan((uav_pos[2] - user_pos[2]) / horizontal_distance)
    return real_distance, theta

def calculate_path_loss(user_pos, uav_pos, A, a, b, B):
    real_distance, theta_rad = calculate_distance_and_theta(user_pos, uav_pos)
    theta_deg = 180 / 3.14159 * theta_rad
    path_loss_angle_part = A / (1 + a * tf.exp(-b * (theta_deg - a)))
    path_loss_distance_part = 20 * tf.math.log(real_distance) / tf.math.log(10.0)  # TensorFlow log base 10
    path_loss = path_loss_angle_part + path_loss_distance_part + B
    return -path_loss

# Modify knowledge_conditioned_layer to use the channel model
def knowledge_conditioned_layer(input_layer):
    def conditioned(x):
        # Here, A, a, b, B are parameters of your model
        A = 1.0
        a = 9.61
        b = 0.16
        B = 20

        # Calculate the path loss for each pair of user and UAV positions
        path_loss = tf.map_fn(lambda p: calculate_path_loss(p[:3], p[3:], A, a, b, B), x)

        # 打印path_loss形状调试
        print(f"Shape of path_loss: {path_loss.shape}")

        # 在返回之前确保输出是二维的
        path_loss = tf.reshape(path_loss, (tf.shape(path_loss)[0], 1))  # 确保它是一个(batch_size, 1)的形状

        return path_loss

    return Lambda(conditioned)(input_layer)

def load_and_scale_data():
    inputs = np.load('../generated_data/qs.npy')
    true_channel_gains = np.load('../generated_data/channel_gains.npy')
    environment_features = np.load('all_structures.npy').astype(np.float32)

    # Normalize environment features
    environment_features[:, :, :, 0] /= 1000
    environment_features[:, :, :, 1] /= 1000
    environment_features[:, :, :, 2] /= 250

    gains_min = np.min(true_channel_gains)
    gains_max = np.max(true_channel_gains)
    true_channel_gains = (true_channel_gains - gains_min) / (gains_max - gains_min)

    num_samples = inputs.shape[0]  # 确保num_samples为1906或你的样本数

    # 使用tile扩展环境特征，使其匹配num_samples
    environment_scaled = np.tile(environment_features, (num_samples, 1,1, 1, 1))  # 修改扩展方式

    return inputs, true_channel_gains, environment_scaled


# Build residual block
def res_block(x, units, apply_knowledge=False):
    # 如果输入的维度大于2，将其展平为二维
    if len(x.shape) > 2:
        x = Flatten()(x)  # 将输入展平

    # 确保形状是定义好的
    if x.shape[-1] is None:
        raise ValueError(f"Input to Dense layer has an undefined last dimension: {x.shape}")

    shortcut = Dense(units)(x)  # Adjust the shortcut dimensions
    x = Dense(units, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)

    print(f"Shape before knowledge conditioning (if applied): {x.shape}")  # 打印形状

    if apply_knowledge:
        x = knowledge_conditioned_layer(x)  # Apply the knowledge module here
        print(f"Shape after knowledge conditioning: {x.shape}")  # 打印形状

    x = Dense(units, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])

    return x



# Build model
def build_model(input_dim, environment_shape):
    input_layer = Input(shape=(input_dim,))
    environment_input = Input(shape=environment_shape)

    # 环境特征编码与知识集成
    environment_encoder = build_environment_encoder(environment_shape)
    encoded_environment = environment_encoder(environment_input)

    # 拼接输入层与环境编码
    concatenated = Concatenate()([input_layer, encoded_environment])

    # 在全连接层之前，检查输入的形状是否正确
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(concatenated)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = res_block(x, 256, apply_knowledge=False)  # 残差块应用于这里
    x = Dropout(0.1)(x)
    x = res_block(x, 128, apply_knowledge=False)
    x = Dropout(0.1)(x)
    x = res_block(x, 64, apply_knowledge=False)
    x = Dropout(0.1)(x)
    x = res_block(x, 32, apply_knowledge=True)
    x = Dropout(0.1)(x)

    x = Dense(1)(x)

    model = Model(inputs=[input_layer, environment_input], outputs=x)
    return model


# Build environment encoder
def build_environment_encoder(environment_shape):
    input_env = Input(shape=environment_shape)
    flattened_env = Flatten()(input_env)
    x = Dense(128, activation='relu')(flattened_env)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation='relu')(x)
    model = Model(inputs=input_env, outputs=x)
    return model

# Main function
def main():
    np.random.seed(None)
    tf.random.set_seed(None)
    # Split data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(inputs_with_knowledge, gains_scaled, test_size=0.2, random_state=42, shuffle=True)
    X_train = np.load('X_train.npy')

    X_test = np.load('X_test.npy')

    y_train = np.load('y_train.npy')

    y_test = np.load('y_test.npy')

    env_train = np.load('env_train.npy')

    env_test = np.load('env_test.npy')


    model = build_model(input_dim=X_train.shape[1], environment_shape=env_train.shape[1:])
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='MSE')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    history = model.fit(
        [X_train, env_train], y_train,
        epochs=500,
        batch_size=32,
        validation_split=0.3,
        callbacks=[early_stopping, lr_scheduler]
    )

    # model.save('kdml_model_with_knowledge_in_hidden_layer.h5')

    y_pred = model.predict([X_test, env_test])
    plt.plot(y_test, 'g', label='True')
    plt.plot(y_pred, 'r', label='Predicted')
    plt.legend()
    plt.show()

    mse = tf.reduce_mean(tf.square(y_pred - y_test)).numpy()
    print(f'Mean Squared Error: {mse:.4f}')

if __name__ == '__main__':
    main()
