from stable_baselines3 import A2C  # 使用A2C替换PPO
from stable_baselines3.common.env_util import make_vec_env
import datetime
import os
import sys
import time
import numpy as np
import tensorflow as tf
import MyLogger
import uav_gym
import uav_gym.envs.uav_env
from KDML_CKM import calculate_path_loss

def main():
    order = '/00/'
    # 保存时间
    year = datetime.datetime.now().year
    month = datetime.datetime.now().month
    day = datetime.datetime.now().day
    year = str(year)
    if month < 10:
        month = '0' + str(month)
    else:
        month = str(month)
    if day < 10:
        day = '0' + str(day)
    else:
        day = str(day)
    date = year + month + day + '/' + order
    date = '20241107' + '/00/'
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print('GPU 使用已启用')
        except RuntimeError as e:
            print('GPU 设置失败:', e)
    else:
        print('未找到 GPU，使用 CPU')

    env = make_vec_env('uav_env-v0', n_envs=8)
    print(env)
    topConfig = uav_gym.envs.uav_env.UavConfig(train_flag=1)
    tensorboard_log = 'user/uav/' + date + 'data/' + \
                      'user%d_dis%d/' % (topConfig.iotd_num, topConfig.total_x) + 'tensorboard/'
    model_save_path = 'user/uav/' + date + 'data/' + \
                      'user%d_dis%d/' % (topConfig.iotd_num, topConfig.total_x) + 'model/' + 'uav_a2c'
    train_log_name = 'user/uav/' + date + 'data/' + \
                     'user%d_dis%d/' % (topConfig.iotd_num, topConfig.total_x) + 'train_log_file'
    if not os.path.exists(train_log_name):
        os.makedirs(train_log_name)
    train_log_name += '/log_file.txt'

    if not os.path.exists(tensorboard_log):
        os.makedirs(tensorboard_log)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    sys.stdout = MyLogger.Logger(train_log_name)
    start_time = time.time()
    # 创建一个A2C算法模型
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    model = A2C('MlpPolicy', env, n_steps=topConfig.t_max, learning_rate=1e-4*2, tensorboard_log=tensorboard_log,
                 gamma=0.99, policy_kwargs=policy_kwargs, verbose=2)
    # 训练
    model.learn(total_timesteps=topConfig.t_max * topConfig.train_time * 8)
    model.save(model_save_path)
    end_time = time.time()
    run_time = end_time - start_time
    print('train finish and the time of training progress is {}s'.format(run_time))


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 只输出错误信息
    main()
