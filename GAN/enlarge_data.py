import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 数据生成器和判别器的超参数
latent_dim = 4
data_dim = 7  # 根据你的数据维度设置
n_critic = 2  # 判别器训练的次数
lambda_gp = 10  # WGAN-GP中的梯度惩罚系数

# 定义生成器
def build_generator():
    model = tf.keras.Sequential([
        Dense(128, input_dim=latent_dim, kernel_initializer='he_normal'),
        LeakyReLU(alpha=0.2),
        BatchNormalization(),
        Dense(256),
        LeakyReLU(alpha=0.2),
        BatchNormalization(),
        Dense(data_dim, activation='linear')
    ])
    return model

# 定义判别器（在WGAN-GP中称为Critic）
def build_critic():
    model = tf.keras.Sequential([
        Dense(64, input_dim=data_dim),
        LeakyReLU(alpha=0.2),
        Dense(32),
        LeakyReLU(alpha=0.2),
        Dense(1, activation='linear')  # 没有sigmoid激活函数
    ])
    return model

# 定义策略网络，用于生成优化噪声
def build_policy_network():
    model = tf.keras.Sequential([
        Dense(64, input_dim=latent_dim, activation='relu'),
        Dense(32, activation='relu'),
        Dense(latent_dim, activation='linear')  # 输出为优化后的噪声
    ])
    return model

# 定义梯度惩罚
# 定义梯度惩罚
def gradient_penalty(critic, real_samples, fake_samples):
    alpha = tf.random.uniform([real_samples.shape[0], 1], 0.0, 1.0, dtype=tf.float32)  # 确保alpha是float32
    real_samples = tf.cast(real_samples, tf.float32)  # 确保real_samples是float32
    fake_samples = tf.cast(fake_samples, tf.float32)  # 确保fake_samples是float32
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    with tf.GradientTape() as tape:
        tape.watch(interpolates)
        validity = critic(interpolates)
    gradients = tape.gradient(validity, [interpolates])[0]
    gradients = tf.reshape(gradients, [tf.shape(gradients)[0], -1])
    gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
    gradient_penalty = tf.reduce_mean((gradient_norm - 1.0) ** 2)
    return gradient_penalty


# 创建生成器、判别器、策略网络和WGAN-GP模型
generator = build_generator()
critic = build_critic()
policy_network = build_policy_network()

# 使用Adam优化器
optimizer_g = Adam(learning_rate=0.00005, beta_1=0.5, beta_2=0.9)
optimizer_c = Adam(learning_rate=0.00005, beta_1=0.5, beta_2=0.9)
optimizer_policy = Adam(learning_rate=0.00001)  # 策略网络优化器

# 训练Critic
@tf.function
def train_critic(real_samples, noise):
    fake_samples = generator(noise, training=True)
    with tf.GradientTape() as tape:
        real_loss = -tf.reduce_mean(critic(real_samples, training=True))
        fake_loss = tf.reduce_mean(critic(fake_samples, training=True))
        gp = gradient_penalty(critic, real_samples, fake_samples)
        loss = real_loss + fake_loss + lambda_gp * gp
    grads = tape.gradient(loss, critic.trainable_variables)
    optimizer_c.apply_gradients(zip(grads, critic.trainable_variables))
    return loss

# 训练生成器
@tf.function
def train_generator(noise):
    with tf.GradientTape() as tape:
        fake_samples = generator(noise, training=True)
        loss = -tf.reduce_mean(critic(fake_samples, training=True))
    grads = tape.gradient(loss, generator.trainable_variables)
    optimizer_g.apply_gradients(zip(grads, generator.trainable_variables))
    return loss

# 训练策略网络
@tf.function
def train_policy(real_samples):
    with tf.GradientTape() as tape:
        noise = tf.random.normal([real_samples.shape[0], latent_dim])
        optimized_noise = policy_network(noise, training=True)
        fake_samples = generator(optimized_noise, training=True)
        reward = -tf.reduce_mean(critic(fake_samples, training=True))  # 负的Critic得分作为奖励
    grads = tape.gradient(reward, policy_network.trainable_variables)
    optimizer_policy.apply_gradients(zip(grads, policy_network.trainable_variables))
    return reward

def plot_losses(g_losses, c_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(c_losses, label='Critic Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# 训练WGAN-GP并使用RL优化策略
def train_wgan_gp(epochs, batch_size=64, callbacks=[]):
    qs = np.load('../CKM/qs.npy')[0:1000]
    qs /= np.max(qs, axis=1, keepdims=True)
    np.save('max_qs.npy', np.max(qs, axis=0))

    gs = np.load('../CKM/channel_gains.npy')[0:1000]
    gs /= np.max(gs, axis=1, keepdims=True)
    np.save('max_gs.npy', np.max(gs, axis=0))

    real_data = np.hstack((qs, gs))
    num_batches = real_data.shape[0] // batch_size

    g_losses = []
    c_losses = []
    rewards = []

    for epoch in range(epochs):
        for _ in range(num_batches):
            # 训练Critic (判别器) n_critic 次
            for _ in range(n_critic):
                noise = tf.random.normal([batch_size, latent_dim])
                idx = np.random.randint(0, real_data.shape[0], batch_size)
                real_samples = real_data[idx]
                c_loss = train_critic(real_samples, noise)
                c_losses.append(c_loss)

            # 使用策略网络优化噪声
            reward = train_policy(real_samples)
            rewards.append(reward)

            # 训练生成器
            optimized_noise = policy_network(tf.random.normal([batch_size, latent_dim]))
            g_loss = train_generator(optimized_noise)
            g_losses.append(g_loss)

        print(f"{epoch}/{epochs} [C loss: {c_loss.numpy()}] [G loss: {g_loss.numpy()}] [Reward: {reward.numpy()}]")

    # 在训练完成时，保存损失曲线
    plot_losses(g_losses, c_losses)

# 使用回调进行训练
train_wgan_gp(epochs=200, batch_size=64)

# 使用生成器生成新数据
def generate_data(num_samples):
    noise = tf.random.normal([num_samples, latent_dim])
    optimized_noise = policy_network(noise)
    generated_data = generator.predict(optimized_noise)
    return generated_data

# 生成额外的数据
new_data = generate_data(num_samples=1000)
np.save('new_data.npy', new_data)
