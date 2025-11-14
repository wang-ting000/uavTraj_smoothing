import gym
from gym import spaces
import math
import numpy as np
import os
import datetime

class UavConfig:
    def __init__(self,train_flag):
        self.train_flag = train_flag
        self.train_time = 60000
        self.h_min = 250
        self.total_x = 1000
        self.total_y = 1000
        # 是否需要重新设置用户位置呢，如果不设置的话用户位置需要读取上一次训练结束的坐标，这样可能泛化能力没这么好了吧
        self.iotd_num = 15
        self.UAV_NUM = 1
        self.UAV_INITIAL_POSITION_X = 0
        self.UAV_INITIAL_POSITION_Y = 0
        self.UAV_INITIAL_POSITION_Z = self.h_min
        self.UAV_INFO_VEC = 6  # [x, y, z, uav_v，0,0] # 保持维度一致
        self.STATE_DIM = self.UAV_INFO_VEC + 8 * self.iotd_num
        self.ACTION_DIM = 4
        self.ACTION_BOUND = [-1, 1]
        # 时隙
        self.t_max = len(np.load(r'C:\Users\Lenovo\PycharmProjects\uavTraj_smoothing\evaluate\plot\pos.npy'))
        # 无人机距离的最大值和速度最大值
        self.V_MAX = 50
        self.V_A_MAX = 20
        self.P = 0.5
        self.R_min = 36

        # 无人机数据 -------------------------------------------------------------------
        # 选择的功率最大值
        self.P_MAX = 26  # dBm
        self.P_min = -70  # dBm
        # NLOS非视距参数和LOS视距参数
        self.area = 'urban'
        if self.area == 'suburban':
            # suburban
            self.LOS = 0.1
            self.NLOS = 21
        elif self.area == 'urban':
            # urban
            self.LOS = 1
            self.NLOS = 20
        elif self.area == 'Dense urban':
            # Dense urban
            self.LOS = 1.6
            self.NLOS = 23
        elif self.area == 'Highrise urban':
            # Highrise urban
            self.LOS = 2.3
            self.NLOS = 34
        # A 计算
        self.A = self.LOS - self.NLOS
        #
        self.c = 3 * (10 ** 8)
        # noise
        self.a = 9.61
        self.b = 0.16
        self.BindWidth = 1
        self.fc = 2000 * (10 ** 6)
        self.B = 20 * np.log10(4 * np.pi * self.fc / self.c) + self.NLOS
        self.N0 = 10 ** (-104 / 10) * 1e-3
        # 保存系数 ------------------------------------------------------------------------
        self.save_step = 500
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
        date = year + month + day + '/'
        date = '20231208' + '/00/'
        if self.train_flag == 1:
            self.save_step = 500
            self.save_model_com_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/com'
            self.save_model_power_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/power'
            self.save_model_track_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/track'
            self.save_model_reward_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/reward'
            self.save_model_com_num_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/com_num'
            self.save_model_r_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/r'
            self.save_model_payload_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/payload'
            self.save_model_endt_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/endt'
            self.save_model_uavv_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/uavv'
            self.save_model_uava_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/uava'
            self.save_model_uav_theta_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/uav_theta'
            self.save_model_uav_elevation_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/uav_elevation'
            self.save_model_reward1_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/every_reward/reward1'
            self.save_model_reward2_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/every_reward/reward2'
            self.save_model_reward3_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/every_reward/reward3'
            self.save_model_reward4_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/every_reward/reward4'
            self.save_model_reward5_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/every_reward/reward5'
            self.save_model_reward6_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/every_reward/reward6'
            self.save_model_reward7_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/every_reward/reward7'
        elif train_flag == 0:
            self.save_step = 100
            self.save_model_com_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/com'
            self.save_model_power_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/power'
            self.save_model_track_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/track'
            self.save_model_reward_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/reward'
            self.save_model_com_num_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/com_num'
            self.save_model_r_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/r'
            self.save_model_payload_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/payload'
            self.save_model_endt_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/endt'
            self.save_model_uavv_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/uavv'
            self.save_model_uava_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/uava'
            self.save_model_reward1_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/every_reward/reward1'
            self.save_model_reward2_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/every_reward/reward2'
            self.save_model_reward3_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/every_reward/reward3'
            self.save_model_reward4_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/every_reward/reward4'
            self.save_model_reward5_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/every_reward/reward5'
            self.save_model_reward6_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/every_reward/reward6'
            self.save_model_reward7_path = 'user1/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/every_reward/reward7'
        if train_flag == 0 or train_flag == 1:
            if not os.path.exists(self.save_model_com_path):
                os.makedirs(self.save_model_com_path)
            if not os.path.exists(self.save_model_power_path):
                os.makedirs(self.save_model_power_path)
            if not os.path.exists(self.save_model_track_path):
                os.makedirs(self.save_model_track_path)
            if not os.path.exists(self.save_model_reward_path):
                os.makedirs(self.save_model_reward_path)
            if not os.path.exists(self.save_model_com_num_path):
                os.makedirs(self.save_model_com_num_path)
            if not os.path.exists(self.save_model_r_path):
                os.makedirs(self.save_model_r_path)
            if not os.path.exists(self.save_model_payload_path):
                os.makedirs(self.save_model_payload_path)
            if not os.path.exists(self.save_model_endt_path):
                os.makedirs(self.save_model_endt_path)
            if not os.path.exists(self.save_model_uavv_path):
                os.makedirs(self.save_model_uavv_path)
            if not os.path.exists(self.save_model_uava_path):
                os.makedirs(self.save_model_uava_path)
            if not os.path.exists(self.save_model_uav_theta_path):
                os.makedirs(self.save_model_uav_theta_path)
            if not os.path.exists(self.save_model_uav_elevation_path):
                os.makedirs(self.save_model_uav_elevation_path)
            if not os.path.exists(self.save_model_reward1_path):
                os.makedirs(self.save_model_reward1_path)
            if not os.path.exists(self.save_model_reward2_path):
                os.makedirs(self.save_model_reward2_path)
            if not os.path.exists(self.save_model_reward3_path):
                os.makedirs(self.save_model_reward3_path)
            if not os.path.exists(self.save_model_reward4_path):
                os.makedirs(self.save_model_reward4_path)
            if not os.path.exists(self.save_model_reward5_path):
                os.makedirs(self.save_model_reward5_path)
            if not os.path.exists(self.save_model_reward6_path):
                os.makedirs(self.save_model_reward6_path)
            if not os.path.exists(self.save_model_reward7_path):
                os.makedirs(self.save_model_reward7_path)
        self.h_max = 750
        self.v_normalize_param = self.V_MAX  # 水平速度最大值
        self.dis_normalize_param = np.sqrt(
            np.square(self.total_x) + np.square(self.total_y) + np.square(self.h_max))  # 最大距离
        self.uav_iotd_com_flag_param = 1
        self.uav_iotd_com_num_param = self.iotd_num
        self.t_param = self.t_max
        self.uav_elevation_param = np.pi / 2

class uavEnv1(gym.Env):
    def __init__(self):
            # 参数
            self.step_i = 0
            self.gap = 10
            self.train_flag = 1

            # 顶层参数
            self.uavConfig = UavConfig(train_flag=self.train_flag)
            self.action_dim = self.uavConfig.ACTION_DIM
            self.STATE_DIM = self.uavConfig.STATE_DIM
            # 定义action_space 和 obs space
            low = 0.
            high = 1.
            s_low = np.array([low] * self.STATE_DIM)
            s_high = np.array([high] * self.STATE_DIM)
            a_low = np.array([-1,0,-1,0])
            a_high = np.array([1,1,1,1])
            self.action_space = spaces.Box(low=a_low, high=a_high, dtype=np.float32)
            self.observation_space = spaces.Box(low=s_low, high=s_high, dtype=np.float32)
            self.h_min = self.uavConfig.h_min
            self.total_x = self.uavConfig.total_x
            self.total_y = self.uavConfig.total_y
            # uav的数量和初始位置
            self.uav_x_pos_init = self.uavConfig.UAV_INITIAL_POSITION_X
            self.uav_y_pos_init = self.uavConfig.UAV_INITIAL_POSITION_Y
            self.uav_z_pos_init = self.uavConfig.UAV_INITIAL_POSITION_Z
            # uav的实时位置
            self.uav_x_pos = self.uav_x_pos_init
            self.uav_y_pos = self.uav_y_pos_init
            self.uav_z_pos = self.uav_z_pos_init
            # self.ep
            self.ep = 0
            # 用户数量和位置
            self.iotd_num = self.uavConfig.iotd_num

            self._reset_iotd_pos()

            # 时隙总时长
            self.t_max = self.uavConfig.t_max
            # 无人机的最大移动距离和最大功率
            self.p_max = self.uavConfig.P_MAX
            self.p_min = self.uavConfig.P_min
            # 无人机运动参数
            self.v_max = self.uavConfig.V_MAX
            self.v_a_max = self.uavConfig.V_A_MAX

            # 关于向量维度的设置
            # action dim 和 state dim 和 action bound
            self.UAV_INFO_VEC = self.uavConfig.UAV_INFO_VEC  # [x, y, z, uav_v, uav_dir, uav_v_z, dir_z]
            self.ACTION_DIM = self.uavConfig.ACTION_DIM  # [uav_v, uav_dir, dir_z, power]
            self.ACTION_BOUND = self.uavConfig.ACTION_BOUND
            # 计算r的参数
            self.A = self.uavConfig.A
            self.N0 = self.uavConfig.N0
            self.a = self.uavConfig.a
            self.b = self.uavConfig.b
            self.B = self.uavConfig.B
            self.BindWidth = self.uavConfig.BindWidth
            # 定义保存无人机信息的矢量
            self.uav_state_normalization = np.zeros(self.STATE_DIM)
            self.uav_action = np.zeros(self.ACTION_DIM)  # 固定选择最近的用户
            # 无人机通信状态
            self.uav_iotd_check_com_bool = np.zeros(self.iotd_num, dtype=np.bool)
            self.uav_iotd_com_time = np.zeros(self.iotd_num)
            # 无人机运动状态
            self.uav_v = 0.0  # 无人机的速度和uav的
            self.uav_a = 0.0
            self.uav_vx = 0.0
            self.uav_vy = 0.0
            self.uav_vz = 0.0
            self.R_min = self.uavConfig.R_min
            self.payload = np.ones((self.t_max,self.iotd_num)) * self.R_min
            # 定义无人机与哪个用户通信和用户通信功率
            self.choose_iotd_power = 0.0
            # 定义无人机与所有用户的距离
            self.uav_iotd_horizontal_distance = np.zeros(self.iotd_num)
            self.uav_iotd_real_distance = np.zeros(self.iotd_num)
            self.uav_iotd_theta = np.zeros(self.iotd_num)
            self.uav_iotd_theta_temp = np.zeros(self.iotd_num)
            self.uav_iotd_path_loss = np.zeros(self.iotd_num)
            self.iotd_receive_power = np.zeros(self.iotd_num)
            # 归一化参数(各项标准的最大值)
            self.v_normalize_param = self.uavConfig.v_normalize_param  # 水平速度最大值
            self.dis_normalize_param = self.uavConfig.dis_normalize_param  # 最大距离
            self.uav_iotd_com_flag_normalization_param = self.uavConfig.uav_iotd_com_flag_param
            self.uav_iotd_com_num_normalization_param = self.uavConfig.uav_iotd_com_num_param
            self.t_normalization_param = self.uavConfig.t_max
            self.elevation_normalize_param = self.uavConfig.uav_elevation_param
            # 路径
            self.save_step = self.uavConfig.save_step
            self.save_model_com_path = self.uavConfig.save_model_com_path
            self.save_model_power_path = self.uavConfig.save_model_power_path
            self.save_model_track_path = self.uavConfig.save_model_track_path
            self.save_model_reward_path = self.uavConfig.save_model_reward_path
            # self.save_model_orignal_reward_path = self.uavConfig.save_model_orignal_reward_path
            self.save_model_com_num_path = self.uavConfig.save_model_com_num_path
            self.save_model_R = self.uavConfig.save_model_r_path
            self.save_model_payload = self.uavConfig.save_model_payload_path
            self.save_model_endt = self.uavConfig.save_model_endt_path
            self.save_model_uavv = self.uavConfig.save_model_uavv_path
            self.save_model_uava = self.uavConfig.save_model_uava_path
            # reward1 reward2 reward3 reward4 reward5 reward6
            self.save_model_reward1_path = self.uavConfig.save_model_reward1_path
            self.save_model_reward2_path = self.uavConfig.save_model_reward2_path
            self.save_model_reward3_path = self.uavConfig.save_model_reward3_path
            self.save_model_reward4_path = self.uavConfig.save_model_reward4_path
            self.save_model_reward5_path = self.uavConfig.save_model_reward5_path
            self.save_model_reward6_path = self.uavConfig.save_model_reward6_path
            self.save_model_reward7_path = self.uavConfig.save_model_reward7_path

            # 初始化所有回合的记录
            self._reset_record_all_eposide()
            # 清理所有无人机实时变量
            self._reset_uav_realtime_state()

    def _reset_record_all_eposide(self):
        # 初始化一个ep
        self.ep = 0
        self.ep_reward = 0.0
        self.all_ep_reward = []
        self.all_ep_com_num = []
        self.reward1_ep = 0.0
        self.reward2_ep = 0.0
        self.reward3_ep = 0.0
        self.reward4_ep = 0.0
        self.reward5_ep = 0.0
        self.reward6_ep = 0.0
        self.all_ep_reward1 = []
        self.all_ep_reward2 = []
        self.all_ep_reward3 = []
        self.all_ep_reward4 = []
        self.all_ep_reward5 = []
        self.all_ep_reward6 = []
        self.all_ep_reward7 = []
        self.all_ep_endt = []

    def _reset_uav_realtime_state(self):
        self.uav_v = 0.0
        self.uav_a = 0.0
        self.p = 0
        self.p_r = 0
        self.p_l = []
        self.reward = 0
        self.uav_x_pos = self.uav_x_pos_init
        self.uav_y_pos = self.uav_y_pos_init
        self.uav_z_pos = self.uav_z_pos_init

    def reset(self):
        self.ep += 1
        self._reset_t()
        # 初始化用户的位置
        self._reset_iotd_pos()
        # 清理无人机所有实时信息
        self._reset_uav_realtime_state()
        # 初始化无人机选择用户和选择用户的功率
        self._reset_choose_user_and_power()
        # 初始化用户的连接情况和记录
        self._reset_uav_iotd_com()
        # 清零标志关系
        self._reset_uav_iotd_com_mark()
        # 清理掉距离角度
        self._reset_theta_distance_pathloss()
        # 更新距离和theta角
        self._calculate_disance_and_theta()
        # 清理R和a
        self._reset_save_uav_R_and_a()
        # 更新state和更新归一化后的state
        self._update_uav_state()
        # 清零所有回合更新变量
        self._reset_record_one_eposide()
        # 清理所有flag
        self._reset_all_flag()
        self.midp = []
        # 获得初始化的状态
        state = self.uav_state_normalization
        return np.array(state)

    def step(self, action):
        # 0、更新时隙
        self._refresh_t()
        # 1、选择速度
        v = np.load(r'C:\Users\Lenovo\PycharmProjects\uavTraj_smoothing\evaluate\plot\v.npy')
        self.uav_v = v[self.step_cnt] if self.step_cnt != self.t_max else v[self.step_cnt-1]
        # 2、选择贝塞尔曲线的系数
        self.p = self._choose_p(action)
        self.p_l.append(self.p)
        # 3、根据系数生成这一段的贝塞尔曲线
        self.new_points = self._genPoints()
        self.p_r = self.p
        # 4、求这一段曲线的点坐标
        self.new_l = self.cal_Bezier(self.new_points)
        # 计算这段曲线的长度
        self.length = self.cal_length(self.new_l)
        self.all_t += self.length / self.uav_v
        # 以这10个点为准，计算这些点的吞吐量
        # 5、
        # 6、计算新的距离和角度
        self.choose_iotd_num_t = []
        self.choose_iotd_num = np.ones((self.gap, self.iotd_num), dtype=np.int64) * 100
        for self.step_i in range(self.gap):
            self.uav_x_pos = self.new_l[self.step_i][0]
            self.uav_y_pos = self.new_l[self.step_i][1]
            self.uav_z_pos = self.new_l[self.step_i][2]
            self._calculate_disance_and_theta()
            # 7、选择用户功率
            self.choose_iotd_power = self._choose_iotd_power(action)
            # 8、选择用户并且标记用户
            self._calculate_all_iotd_receve_power()
            self.choose_iotd_num_t.append(self._choose_all_iotd_in_range())
            self._mark_uav_iotd_com()
            # 9、计算r与用户进行通信
            self._calculate_iotd_per_R()
            # 11、获得新的状态
            self._update_uav_state()  # 更新归一化state
            save_uav_track_temp = np.hstack((self.uav_x_pos, self.uav_y_pos, self.uav_z_pos))
            self.save_uav_track.append(save_uav_track_temp)
            # 13、记录轨迹和通信记录，用来后续查看
        self._save_uav_track_and_com()
        self.reward = self._reward_function()
        # 10、获取是否完成
        self.uav_finish_com_task_flag, self.terminated = self._check_done()

        # 12、根据上面的action和是否完成计算r


        # 14、保存reward
        self._save_uav_epreward()
        # self.determine_convergence()
        # 15、返回变量
        state = self.uav_state_normalization
        reward = self.reward
        terminated = self.terminated
        return np.array(state), reward, terminated, {}

    def _save_uav_epreward(self):
        if self.terminated:
            # self.all_ep_orignal_reward.append(self.ep_reward)
            self.all_ep_com_num.append(self.uav_iotd_com_num)
            # reward1 reward2 reward3 reward4 reward5 reward6
            self.all_ep_reward1.append(self.reward1_ep)
            self.all_ep_reward2.append(self.reward2_ep)
            self.all_ep_reward3.append(self.reward3_ep)
            self.all_ep_reward4.append(self.reward4_ep)
            self.all_ep_reward5.append(self.reward5_ep)
            self.all_ep_reward6.append(self.reward6_ep)
            self.all_ep_reward7.append(self.reward7_ep)
            self.all_ep_reward.append(self.ep_reward)

            if self.step_cnt==self.t_max and self.ep == self.uavConfig.train_time:
                save_uav_reward_name = os.path.join(self.save_model_reward_path, 'uav_reward')
                # save_uav_orignal_reward_name = os.path.join(self.save_model_orignal_reward_path, 'uav_orignal_reward_%d' % (self.ep))
                save_endt_name = os.path.join(self.save_model_endt, 'endt')
                np.save(save_uav_reward_name, self.all_ep_reward)
                # np.save(save_uav_orignal_reward_name, self.all_ep_orignal_reward)
                np.save(save_endt_name, self.all_ep_endt)
                np.save('allt',self.all_t)

                # reward1 reward2 reward3 reward4 reward5 reward6
                save_uav_reward1_name = os.path.join(self.save_model_reward1_path, 'uav_reward1')
                save_uav_reward2_name = os.path.join(self.save_model_reward2_path, 'uav_reward2')
                save_uav_reward3_name = os.path.join(self.save_model_reward3_path, 'uav_reward3')
                save_uav_reward4_name = os.path.join(self.save_model_reward4_path, 'uav_reward4')
                save_uav_reward5_name = os.path.join(self.save_model_reward5_path, 'uav_reward5')
                save_uav_reward6_name = os.path.join(self.save_model_reward6_path, 'uav_reward6')
                save_uav_reward7_name = os.path.join(self.save_model_reward7_path, 'uav_reward7')

                np.save(save_uav_reward1_name, self.all_ep_reward1)
                np.save(save_uav_reward2_name, self.all_ep_reward2)
                np.save(save_uav_reward3_name, self.all_ep_reward3)
                np.save(save_uav_reward4_name, self.all_ep_reward4)
                np.save(save_uav_reward5_name, self.all_ep_reward5)
                np.save(save_uav_reward6_name, self.all_ep_reward6)
                np.save(save_uav_reward7_name, self.all_ep_reward7)

    def _save_uav_track_and_com(self):

        # save_uav_track_temp = np.hstack((self.uav_x_pos, self.uav_y_pos, self.uav_z_pos))
        # self.save_uav_track.append(save_uav_track_temp)
        # self.save_uav_iotd_com.append(self.choose_iotd_num)
        self.save_uav_iotd_power.append(np.array(self.choose_iotd_power))
        self.save_uav_v.append(self.uav_v)
        # self.save_uav_a.append(self.uav_v_a)
        self.save_uav_iotd_com.append(self.choose_iotd_num)

        if self.step_cnt == self.t_max and self.ep == self.uavConfig.train_time:
            np.save('p_l', self.p_l)
            save_uav_track_name = os.path.join(self.save_model_track_path, 'uav_track')
            save_uav_iotd_com_name = os.path.join(self.save_model_com_path, 'uav_iotd_com')
            save_uav_iotd_com_power_name = os.path.join(self.save_model_power_path,
                                                        'uav_iotd_com_powe')
            save_iotd_model_payload_name = os.path.join(self.save_model_payload, 'iotd_payload')
            save_model_R_name = os.path.join(self.save_model_R, 'R')
            save_uav_v_name = os.path.join(self.save_model_uavv, 'uav_v')
            save_uav_a_name = os.path.join(self.save_model_uava, 'uav_a')
            # 保存
            np.save(save_uav_track_name, self.save_uav_track)
            np.save(save_uav_iotd_com_name, np.asarray(self.save_uav_iotd_com, dtype=object))
            np.save(save_uav_iotd_com_power_name, self.save_uav_iotd_power)
            np.save(save_iotd_model_payload_name, self.payload)
            np.save(save_model_R_name, self.R_remain)
            np.save(save_uav_v_name, self.save_uav_v)
            np.save(save_uav_a_name, self.save_uav_a)


    def  _reward_function(self):
        if not self.uav_finish_com_task_flag:
            scale1 = 0.003
        else:
            scale1 = 0
        scale2 = 0.003
        scale3 = 0.0002
        reward1 = 0
        reward2 = -scale2 * self.all_t
        if not self.uav_finish_com_task_flag:
            reward3 = scale3 * self.iotd_com_finish_num * self.all_t
        else:
            reward3 = 0
        reward = 0 + reward2 + reward3

        self.reward1_ep += reward1
        self.reward2_ep += reward2
        self.reward3_ep += reward3
        self.ep_reward += reward


        return reward


    def _check_done(self):
        # 检查是否已经完成通信任务
        self.iotd_com_finish_num = np.sum(self.finish_com_task_flag)
        # 判断是否增加一个用户
        if self.iotd_com_finish_num == self.iotd_com_last_time_finish_num:
            self.add_new_user_flag = False
        else:
            self.add_new_user_flag = True

        self.add_new_num = self.iotd_com_finish_num - self.iotd_com_last_time_finish_num
        self.iotd_com_last_time_finish_num = self.iotd_com_finish_num


        if self.step_cnt == self.t_max:
            terminated = True
        else:
            terminated = False

        if self.iotd_com_finish_num==self.iotd_num:
            uav_finish_com_task_flag = True
        else:
            uav_finish_com_task_flag = False

        return uav_finish_com_task_flag, terminated

    def _calculate_iotd_per_R(self):
        for i in range(self.iotd_num):
            if self.is_com[i] == 1:
                p_r_temp = self.iotd_receive_power[i]
                p_r = 10 ** (p_r_temp / 10) * 1e-3
                sinr = np.true_divide(p_r, self.N0)
                r_temp = self.BindWidth * np.log2(1 + sinr)
                self.R_remain[(self.step_cnt):, i] += r_temp
                if self.R_remain[self.step_cnt][i] >= self.R_min:
                    self.finish_com_task_flag[i] = 1
                    self.payload[(self.step_cnt):, i] = 0.0

                else:
                    self.finish_com_task_flag[i] = 0
                    self.payload[(self.step_cnt):, i] = self.R_min - self.R_remain[self.step_cnt][i]


    # 选择覆盖方位得所有用户
    def _choose_all_iotd_in_range(self):
        choose_num = []
        for i in range(self.iotd_num):
            if self.iotd_receive_power[i] >= self.p_min:
                choose_num.append(i)
        return choose_num

    def _mark_uav_iotd_com(self):
        self.is_com = np.zeros(self.iotd_num)
        for i in self.choose_iotd_num_t[self.step_i]:
            if self.finish_com_task_flag[i]==0:
                self.choose_iotd_num[self.step_i][i] = i
                self.uav_iotd_check_com_bool[i] = 1
                self.is_com[i] = 1


    def _calculate_all_iotd_receve_power(self):
        self.iotd_receive_power = np.array([self.choose_iotd_power] * self.iotd_num) - self.uav_iotd_path_loss


    def _reset_iotd_pos(self):
        a = np.load(r'C:\Users\Lenovo\PycharmProjects\uavTraj_smoothing\evaluate\plot\iotd_pos.npy')
        self.iotd_x_pos = [0] * self.iotd_num
        self.iotd_y_pos = [0] * self.iotd_num
        self.iotd_z_pos = [0] * self.iotd_num
        for i in range(self.iotd_num):
            self.iotd_x_pos[i], self.iotd_y_pos[i], self.iotd_z_pos[i] = a[i]

    def _reset_choose_user_and_power(self):
        # 初始化选择用户的
        self.choose_iotd_power = 0.0

    def _reset_uav_iotd_com(self):
        for i in range(self.iotd_num):
            self.uav_iotd_check_com_bool[i] = 0
            self.uav_iotd_com_time[i] = 0
        self.uav_iotd_com_num = 0

    def _reset_uav_iotd_com_mark(self):
        # 记录与哪个用户进行通信
        self.uav_iotd_check_com_bool = np.zeros(self.iotd_num, dtype=np.bool)
        self.uav_iotd_com_time = np.zeros(self.iotd_num)

        # 初始化路径角度等信息

    def _reset_theta_distance_pathloss(self):
        self.uav_iotd_horizontal_distance = np.zeros(self.iotd_num)
        self.uav_iotd_real_distance = np.zeros(self.iotd_num)
        self.uav_iotd_theta = np.zeros(self.iotd_num)
        self.uav_iotd_theta_temp = np.zeros(self.iotd_num)
        self.uav_iotd_path_loss = np.zeros(self.iotd_num)
        self.iotd_receive_power = np.zeros(self.iotd_num)

        self.uav_iotd_horizontal_distance_s = np.zeros(self.iotd_num)
        self.uav_iotd_real_distance_s = np.zeros(self.iotd_num)
        self.uav_iotd_theta_s = np.zeros(self.iotd_num)
        self.uav_iotd_theta_temp_s = np.zeros(self.iotd_num)
        self.uav_iotd_path_loss_s = np.zeros(self.iotd_num)
        self.iotd_receive_power_s = np.zeros(self.iotd_num)

    def _calculate_disance_and_theta(self):
        for i in range(self.iotd_num):
            self.uav_iotd_horizontal_distance[i] = np.sqrt(np.square(self.uav_x_pos - self.iotd_x_pos[i])
                                                           + np.square(self.uav_y_pos - self.iotd_y_pos[i])) + 1
            self.uav_iotd_real_distance[i] = np.sqrt(np.square(self.uav_iotd_horizontal_distance[i])
                                                     + np.square(self.uav_z_pos - self.iotd_z_pos[i]))
            self.uav_iotd_theta[i] = np.arctan(
                np.true_divide(self.uav_z_pos - self.iotd_z_pos[i], self.uav_iotd_horizontal_distance[i]))
            self.uav_iotd_path_loss[i] = np.true_divide(self.A, 1 + self.a * np.exp(
                -self.b * (180 / np.pi * self.uav_iotd_theta[i] - self.a))) + 20 * np.log10(
                self.uav_iotd_horizontal_distance[i] / np.cos(self.uav_iotd_theta_temp[i])) + self.B

    # 清零R和a
    def _reset_save_uav_R_and_a(self):
        # 用来剩余通信量的变量
        self.R_remain = np.zeros((self.t_max,self.iotd_num))
        self.finish_com_task_flag = np.zeros(self.iotd_num)


    def _update_uav_state(self):
        self.uav_state_normalization[0] = self.uav_x_pos / self.total_x
        self.uav_state_normalization[1] = self.uav_y_pos / self.total_y
        self.uav_state_normalization[2] = self.uav_z_pos / self.h_min
        self.uav_state_normalization[3] = self.uav_v / self.v_normalize_param
        self.uav_state_normalization[4] = 0
        self.uav_state_normalization[5] = 0
        for i in range(self.iotd_num):
            self.VEC = self.uavConfig.UAV_INFO_VEC
            self.uav_state_normalization[self.VEC + 8 * i] = self.iotd_x_pos[i] / self.total_x
            self.uav_state_normalization[self.VEC + 8 * i + 1] = self.iotd_y_pos[i] / self.total_y
            self.uav_state_normalization[self.VEC + 8 * i + 2] = self.iotd_z_pos[i] / self.h_min
            self.uav_state_normalization[self.VEC + 8 * i + 3] = self.uav_iotd_check_com_bool[i] / 1.0
            self.uav_state_normalization[self.VEC + 8 * i + 4] = self.finish_com_task_flag[i] / 1.0
            self.uav_state_normalization[self.VEC + 8 * i + 5] = self.uav_iotd_theta[i] / self.elevation_normalize_param
            self.uav_state_normalization[self.VEC + 8 * i + 6] = self.uav_iotd_real_distance[
                                                                     i] / self.dis_normalize_param
            self.uav_state_normalization[self.VEC + 8 * i + 7] = self.payload[self.step_cnt-1][i] / self.R_min

        # 清零回合记录参数

    def _reset_record_one_eposide(self):
        # 用来保存回合的奖励参数
        self.ep_reward = 0.
        self.reward1_ep = 0.
        self.reward2_ep = 0.
        self.reward3_ep = 0.
        self.reward4_ep = 0.
        self.reward5_ep = 0.
        self.reward6_ep = 0.
        self.reward7_ep = 0.
        self.reward_t_ep = 0.

        #  用来保存无人机
        self.save_uav_track = []
        self.save_uav_iotd_com = []
        self.save_uav_iotd_power = []
        self.save_iotd_payload = []
        self.save_uav_v = []
        self.save_uav_a = []
        # punishment
        self.punishment = 0.0

    # 清零所有标志位
    def _reset_all_flag(self):
        self.uav_finish_com_task_flag = False
        self.uav_finish_go_back_flag = False
        self.terminated = False
        self.already_give_t_reward_flag = False
        self.ahead_of_time_finish_flag = False
        self.iotd_com_last_time_finish_num = 0
        self.iotd_com_finish_num = 0
        self.add_new_user_flag = False
        self.last_time_distance = -1

    def _reset_t(self):
        self.step_cnt = 0    # 轨迹的段数，也可以看做step的次数
        self.all_t = 0      # 总共飞行时间

    def _refresh_t(self):
        self.step_cnt += 1



# 选择速度
    def _choose_uav_v(self, action, uav_v):
        # 用加速度
        if not self.uav_finish_com_task_flag:
            action_0_temp = np.clip(action[0], -1, 1)
            choose_uav_v_a = action_0_temp * self.v_a_max
            if uav_v == self.v_max and choose_uav_v_a >= 0:
                self.uav_v_a = 0
            elif uav_v == 0 and choose_uav_v_a <= 0:
                self.uav_v_a = 0
            else:
                self.uav_v_a = choose_uav_v_a
            uav_v = uav_v + self.uav_v_a
            if uav_v > self.v_max:
                uav_v = self.v_max
            if uav_v <= 0:
                uav_v = +0.00001
        else:
            action_0_temp = 1.0
            choose_uav_v_a = action_0_temp * self.v_a_max
            if uav_v == self.v_max and choose_uav_v_a >= 0:
                self.uav_v_a = 0
            elif uav_v == 0 and choose_uav_v_a <= 0:
                self.uav_v_a = 0
            else:
                self.uav_v_a = choose_uav_v_a
            uav_v = uav_v + self.uav_v_a
            if uav_v > self.v_max:
                uav_v = self.v_max
            if uav_v <= 0:
                uav_v = +0.00001
        return uav_v

    def _choose_p(self,action):
        p = np.clip(action[3],0,1)
        p = p*0.5
        return p

    def _genPoints(self): #每一段生成midpoints
        oriTraj = np.load(r'C:\Users\Lenovo\PycharmProjects\uavTraj_smoothing\evaluate\plot\pos.npy')
        mid1 = [0,0,0]
        mid2 = [0,0,0]
        mid3 = [0, 0, 0]
        if self.step_cnt == 1: # 第一段是一阶贝塞尔
            p = oriTraj[0]
            q = oriTraj[1]
            points = np.zeros((3,2))
            mid1[0] = self.p*q[0] + (1-self.p)*p[0]
            mid1[1] = self.p * q[1] + (1 - self.p) * p[1]
            mid1[2] = self.p * q[2] + (1 - self.p) * p[2]
            points[0] = [p[0],mid1[0]]
            points[1] = [p[1], mid1[1]]
            points[2] = [p[2], mid1[2]]
            return points
        if self.step_cnt == self.t_max: # 最后一段是二阶
            p = oriTraj[self.step_cnt-2]
            q = oriTraj[self.step_cnt - 1]
            points = np.zeros((3, 3))
            mid1[0] = self.p_r * q[0] + (1 - self.p_r) * p[0]
            mid1[1] = self.p_r * q[1] + (1 - self.p_r) * p[1]
            mid1[2] = self.p_r * q[2] + (1 - self.p_r) * p[2]
            mid2[0] = self.p_r * p[0] + (1 - self.p_r) * q[0]
            mid2[1] = self.p_r * p[1] + (1 - self.p_r) * q[1]
            mid2[2] = self.p_r * p[2] + (1 - self.p_r) * q[2]
            points[0] = [mid1[0],mid2[0],q[0]]
            points[1] = [mid1[1],mid2[1],q[1]]
            points[2] = [mid1[2],mid2[2],q[2]]
            return points
        else:
            p = oriTraj[self.step_cnt - 2]
            q = oriTraj[self.step_cnt - 1]
            r = oriTraj[self.step_cnt]
            points = np.zeros((3, 4))
            mid1[0] = self.p_r * q[0] + (1 - self.p_r) * p[0]
            mid1[1] = self.p_r * q[1] + (1 - self.p_r) * p[1]
            mid1[2] = self.p_r * q[2] + (1 - self.p_r) * p[2]
            mid2[0] = self.p_r * p[0] + (1 - self.p_r) * q[0]
            mid2[1] = self.p_r * p[1] + (1 - self.p_r) * q[1]
            mid2[2] = self.p_r * p[2] + (1 - self.p_r) * q[2]
            mid3[0] = self.p * r[0] + (1 - self.p) * q[0]
            mid3[1] = self.p * r[1] + (1 - self.p) * q[1]
            mid3[2] = self.p * r[2] + (1 - self.p) * q[2]
            points[0] = [mid1[0],mid2[0], q[0], mid3[0]]
            points[1] = [mid1[1], mid2[1], q[1], mid3[1]]
            points[2] = [mid1[2], mid2[2], q[2], mid3[2]]
            return points



    def cal_Bezier(self,p):
        def comb(n, m):
            return (math.factorial(n) / (math.factorial(m) * math.factorial(n - m)))

        B = np.zeros((self.gap,3))
        st = 0
        for t in np.linspace(0,1,self.gap,endpoint=True):
            if self.step_cnt == 1:
                nn = 1
            elif self.step_cnt == self.t_max:
                nn = 2
            else:
                nn = 3
            for i in range(nn + 1):
                B[0 + st][0] += comb(nn, i) * p[0][i] * ((1 - t) ** (nn - i)) * (t ** i)
                B[0 + st][1] += comb(nn, i) * p[1][i] * ((1 - t) ** (nn - i)) * (t ** i)
                B[0 + st][2] += comb(nn, i) * p[2][i] * ((1 - t) ** (nn - i)) * (t ** i)
            st += 1
        return B

    def cal_length(self,line):
        length = 0.0

        for i in range(len(line) - 1):
            x1, y1, z1 = line[i]
            x2, y2, z2 = line[i + 1]

            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2+ (z2 - z1) ** 2)
            length += distance
        return length

    # 选择用户功率
    def _choose_iotd_power(self, action):
        action_3_temp = np.clip(action[2], 0, 1)
        choose_iotd_p = action_3_temp * self.p_max
        return choose_iotd_p


