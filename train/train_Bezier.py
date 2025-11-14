'''
作用：主要用来训练的模型的
'''
import datetime
import os
import sys
import time
import numpy as np

import MyLogger
import uav_gym
from uav_gym.envs import uav_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import BasePolicy


class MlpPolicy(BasePolicy):
    def init(self,  *args, **kwargs):
        super(MlpPolicy, self).init(*args, **kwargs,
                                           net_arch=[dict(pi=[256, 256],
                                                          vf=[256, 256])],
                                           feature_extraction="mlp")

# from stable_baselines import DDPG
# tensorboard --logdir=train\user1\uav\20231024\00\data\user15_dis1000\tensorboard\PPO_23 --host=127.0.0.1

# 这里定义了mlp神经网络，中间层是256，256然后通过
# class UavMlpPolicy(FeedForwardPolicy):
#     def __init__(self, *args, **kwargs):
#         super(UavMlpPolicy, self).__init__(*args, **kwargs,
#                                            net_arch=[dict(pi=[256, 256],
#                                                           vf=[256, 256])],
#                                            feature_extraction="mlp")
#
#
# register_policy('UavMlpPolicy', UavMlpPolicy)
#
#
# class UavMlpLstmPolicy(LstmPolicy):
#     def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=128, reuse=False, **_kwargs):
#         super(UavMlpLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
#                                                layer_norm=False, feature_extraction="mlp", **_kwargs)
#
#
# register_policy('UavMlpLstmPolicy', UavMlpLstmPolicy)


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
    date = '20231208' + '/00/'

    env1 = make_vec_env('uav_env1-v0')
    topConfig = uav_gym.envs.uav_env1.UavConfig(train_flag=1)
    tensorboard_log = 'user1/uav/' + date + 'data/' + \
                      'user%d_dis%d/' % (topConfig.iotd_num, topConfig.total_x) + 'tensorboard/'
    model_save_path = 'user1/uav/' + date + 'data/' + \
                      'user%d_dis%d/' % (topConfig.iotd_num, topConfig.total_x) + 'model/' + 'uav_ppo2'
    train_log_name = 'user1/uav/' + date + 'data/' + \
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
    t = np.load(r'C:\Users\Lenovo\PycharmProjects\uavTraj_smoothing\evaluate\plot\pos.npy')
    # 创建一个PPO算法模型
    # model = PPO.load('user/uav/20231024/00/data/user15_dis1000/model/uav_ppo2.zip')
    # model.set_env(env1)
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[256, 256]))
    model = PPO('MlpPolicy', env1, n_steps=topConfig.t_max, learning_rate=1e-5, tensorboard_log=tensorboard_log,
                gamma=0.995, policy_kwargs=policy_kwargs, verbose=1, clip_range=0.20,batch_size=topConfig.t_max)
    # 训练
    # 训练
    model.learn(total_timesteps=len(t) * topConfig.train_time)
    model.save(model_save_path)
    end_time = time.time()
    run_time = end_time - start_time
    print('train finish and the time of training progress is {}s'.format(run_time))


if __name__ == '__main__':
    main()
