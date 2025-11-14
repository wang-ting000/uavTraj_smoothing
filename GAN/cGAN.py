import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 数据生成器和判别器的超参数
latent_dim = 4  # 随机噪声的维度
cond_dim = 5  # 条件向量的维度（假设为5维）
data_dim = 7  # 生成数据的维度


# 定义生成器
def build_generator(latent_dim, cond_dim, data_dim):
    noise_input = Input(shape=(latent_dim,))
    cond_input = Input(shape=(cond_dim,))
    merged_input = Concatenate()([noise_input, cond_input])

    x = Dense(128, kernel_initializer='he_normal')(merged_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    output = Dense(data_dim, activation='linear')(x)

    model = Model([noise_input, cond_input], output)
    return model


# 定义判别器
def build_discriminator(data_dim, cond_dim):
    data_input = Input(shape=(data_dim,))
    cond_input = Input(shape=(cond_dim,))
    merged_input = Concatenate()([data_input, cond_input])

    x = Dense(256)(merged_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.2)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model([data_input, cond_input], output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    return model


# 定义cGAN模型
def build_cgan(generator, discriminator):
    discriminator.trainable = False
    noise_input = Input(shape=(latent_dim,))
    cond_input = Input(shape=(cond_dim,))
    generated_data = generator([noise_input, cond_input])
    validity = discriminator([generated_data, cond_input])

    model = Model([noise_input, cond_input], validity)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001))
    return model


# 创建生成器、判别器和cGAN模型
generator = build_generator(latent_dim, cond_dim, data_dim)
discriminator = build_discriminator(data_dim, cond_dim)
cgan = build_cgan(generator, discriminator)

g_losses = []
d_losses = []


def plot_losses(g_losses, d_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# 训练cGAN的函数
def train_cgan(epochs, batch_size, latent_dim, cond_dim, data_dim, real_data, real_conditions):
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # 随机抽取真实数据样本
        idx = np.random.randint(0, real_data.shape[0], batch_size)
        real_samples = real_data[idx]
        real_cond = real_conditions[idx]

        # 生成虚假数据
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_samples = generator.predict([noise, real_cond])

        # 训练判别器
        d_loss_real = discriminator.train_on_batch([real_samples, real_cond], valid)
        d_loss_fake = discriminator.train_on_batch([gen_samples, real_cond], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # 训练生成器
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = cgan.train_on_batch([noise, real_cond], valid)

        g_losses.append(g_loss)
        d_losses.append(d_loss)

        if epoch % 10 == 0:
            print(f"{epoch}/{epochs} [D loss: {d_loss}] [G loss: {g_loss}]")
            plot_losses(g_losses, d_losses)


# 示例：使用随机生成的真实数据和条件信息
real_data = np.random.rand(1000, data_dim)  # 1000个真实数据样本
real_conditions = np.random.rand(1000, cond_dim)  # 1000个条件向量

# 训练cGAN模型
train_cgan(epochs=200, batch_size=64, latent_dim=latent_dim, cond_dim=cond_dim, data_dim=data_dim, real_data=real_data,
           real_conditions=real_conditions)


# 使用生成器生成新数据
def generate_conditional_data(generator, num_samples, latent_dim, cond_dim, condition):
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    generated_data = generator.predict([noise, condition])
    return generated_data


# 示例：生成特定条件下的新数据
new_condition = np.random.rand(500, cond_dim)  # 500个样本的条件向量
generated_data = generate_conditional_data(generator, num_samples=500, latent_dim=latent_dim, cond_dim=cond_dim,
                                           condition=new_condition)

# 保存生成的数据
np.save('generated_data.npy', generated_data)
