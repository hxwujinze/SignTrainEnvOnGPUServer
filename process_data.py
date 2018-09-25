# coding:utf-8

import os
import pickle

import numpy as np
import pywt
from sklearn import preprocessing

WINDOW_SIZE = 16
TYPE_LEN = {
    'acc': 3,
    'gyr': 3,
    'emg': 8
}

'''
提取一个手势的一个batch的某一信号种类的全部数据
数据形式保存不变 只改变数值和每次采集ndarray的长度
（特征提取会改变数据的数量）
'''

# data process func for online

def feature_extract(data_set, type_name):
    """
    特征提取 并进行必要的归一化

    acc gyr数据的三种特征量纲相差不大 且有某些维度全局的值都很相近的情况
    于是暂时去除归一化的操作 拟对只对数据变化较大，且变化范围较大于1的数据维度进行部分归一化

    emg数据照常进行各种处理

    :param data_set: 来自Load_From_File过程的返回值 一个dict
                     包含一个手语 三种采集数据类型的 多次采集过程的数据
    :param type_name: 数据采集的类型 决定nparray的长度
    :return: 一个dict 包含这个数据采集类型的原始数据,3种特征提取后的数据,特征拼接后的特征向量
            仍保持多次采集的数据的np.array放在一个list中
            返回的数据的dict包含所有的数据 但是只有有效的字段有数据
    """

    data_set_polyfit_feat = []  # for cnn 使用多项式对间隔间的数据进行拟合 减少中间数据点
    data_set_appended_feat = []

    data_set = data_set[type_name]
    for raw_data in range(len(data_set)):
        if raw_data % 100 == 0:
            print("extraction progress %d / %d" % (raw_data, len(data_set)))
        raw_data = data_set[raw_data]

        # cnn的特征提取过程 只使用曲线拟合特征
        seg_polyfit_feat = feature_extract_single_polyfit(raw_data, 2)
        # 多项式拟合后切割
        seg_polyfit_feat = seg_polyfit_feat[8:-8, :]
        # 给CNN喂128的片段短数据  拟合压缩前是
        data_set_polyfit_feat.append(seg_polyfit_feat)


    return {
        'type_name': type_name,
        'raw': data_set,
        'poly_fit': data_set_polyfit_feat,  # cnn 3 x 64 数据
        'append_all': data_set_appended_feat  # rnn 11 x 10 数据
    }

def feature_extract_single_polyfit(data, compress):
    """
    execute the ploy fit smooth and compression on single data mat (acc gyr)
    nparray ( (dim 1) , (dim 2), (dim 3) )
    16 points window and 3-order poly fit
    compress mean take out some point in sequence according to fix length internal,
    likes down sampling

    :param data: data mat contains three channel data
    :param compress: compress ratio, the sampling window
            # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15  rate = 2
            # 0   2   4   6   8   10    11    14
    :return: after fitting data 3 dim, but data len in each dim has changed by compress rate
    """
    seg_poly_fit = None
    window_range = 16
    start_ptr = 0
    end_ptr = window_range
    while end_ptr <= len(data):
        window_data = data[start_ptr:end_ptr, :]
        window_extract_data = None
        x = np.arange(0, window_range, 1)
        y = window_data
        # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
        # 0   2   4   6   8   10    11    14
        poly_args = np.polyfit(x, y, 3)
        for each_channel in range(3):
            dots_in_channel = None
            window_poly = np.poly1d(poly_args[:, each_channel])
            for dot in np.arange(0, window_range, compress):
                # assemble each dot's each channel
                if dots_in_channel is None:
                    dots_in_channel = window_poly(dot)
                else:
                    dots_in_channel = np.vstack((dots_in_channel, window_poly(dot)))
            # assemble each window's each channel data
            if window_extract_data is None:
                window_extract_data = dots_in_channel
            else:
                window_extract_data = np.hstack((window_extract_data, dots_in_channel))

        # assemble each window data
        if seg_poly_fit is None:
            seg_poly_fit = window_extract_data
        else:
            seg_poly_fit = np.vstack((seg_poly_fit, window_extract_data))
        start_ptr += window_range
        end_ptr += window_range

    return seg_poly_fit

def feature_extract_single(input_data, type_name):
    """
    execute ARC RMS ZC feature extraction in single data mat (acc, gyr)
    :param input_data:
    :param type_name:
    :return:
    """
    # 对曲线拟合后的数据进行特征提取 效果更好
    polyfit_data = feature_extract_single_polyfit(input_data, 1)
    window_amount = len(polyfit_data) / WINDOW_SIZE
    windows_data = np.vsplit(polyfit_data, window_amount)
    win_index = 0
    seg_ARC_feat = None
    seg_RMS_feat = None
    seg_ZC_feat = None

    features = [seg_RMS_feat, seg_ZC_feat, seg_ARC_feat, ]
    for Win_Data in windows_data:
        # 依次处理每个window的数据
        win_RMS_feat = np.sqrt(np.mean(np.square(Win_Data), axis=0))
        Win_Data1 = np.vstack((Win_Data[1:, :], np.zeros((1, TYPE_LEN[type_name]))))
        win_ZC_feat = np.sum(np.sign(-np.sign(Win_Data) * np.sign(Win_Data1) + 1), axis=0) - 1
        win_ARC_feat = np.apply_along_axis(ARC, 0, Win_Data)
        # arc 只要后面几个系数的数据
        win_ARC_feat = win_ARC_feat.ravel()[-5:, ]
        # 将每个window特征提取的数据用vstack叠起来
        win_index += 1

        each_feat_type_data = (win_RMS_feat, win_ZC_feat, win_ARC_feat)
        for each in range(len(each_feat_type_data)):
            if features[each] is None:
                features[each] = each_feat_type_data[each]
            else:
                features[each] = np.vstack((features[each], each_feat_type_data[each]))

    seg_all_feat = np.hstack(tuple(features))
    features.append(polyfit_data)
    features.append(seg_all_feat)
    # rms zc arc polyfit all
    return tuple(features)



def append_feature_vector(data_set, with_emg=False):
    """
    拼接三种数据采集类型的特征数据成一个大向量
    :param data_set: 第一维存储三种采集类型数据集的list
                     第二维是这个类型数据三种特征拼接后 每次采集获得的数据矩阵
    :param with_emg: 是否将emg也拼接进入向量
    :return:
    """

    batch_list = []
    # 每种采集类型下有多个数据
    for i in range(len(data_set[0])):
        # 取出每个采集类型的数据列中的每个数据进行拼接
        if with_emg:
            batch_mat = append_single_data_feature(acc_data=data_set[0][i],
                                                   gyr_data=data_set[1][i],
                                                   emg_data=data_set[2][i], )
        else:
            batch_mat = append_single_data_feature(acc_data=data_set[0][i],
                                                   gyr_data=data_set[1][i],)

        batch_list.append(batch_mat)
    return batch_list

def append_single_data_feature(acc_data, gyr_data, emg_data=None):
    batch_mat = np.zeros(len(acc_data))
    is_first = True
    for each_window in range(len(acc_data)):
        # 针对每个识别window
        # 把这一次采集的三种数据采集类型进行拼接
        line = np.append(acc_data[each_window], gyr_data[each_window])
        if emg_data is not None:
            line = np.append(line, emg_data[each_window])
        if is_first:
            is_first = False
            batch_mat = line
        else:
            batch_mat = np.vstack((batch_mat, line))
    return batch_mat

# emg data_process

def emg_feature_extract(data_set, expanded):
    """
    特征提取
    :param data_set: 来自Load_From_File过程的返回值 一个dict
                     包含一个手语 三种采集数据类型的 多次采集过程的数据
    :param expanded: 是否进行上采样，将其变为与acc gyr数据同样长度
    :return: 一个dict 包含这个数据采集类型的原始数据,3种特征提取后的数据,特征拼接后的特征向量
            仍保持多次采集的数据放在一起
    """
    data_set = [each[16:144, :] for each in data_set['emg']]
    data_trans = emg_wave_trans(data_set)
    if expanded:
        data_trans = expand_emg_data(data_trans)
    return {
        'type_name': 'emg',
        'raw': data_set,
        'trans': data_trans,
    }

def wavelet_trans(data):
    data = np.array(data).T  # 转换为 通道 - 时序
    data = pywt.threshold(data, 30, 'hard')  # 阈值滤波
    if len(data[0]) == 160:

        data = pywt.wavedec(data, wavelet='db2', level=5)
        data = np.vstack((data[0].T, np.zeros(8)))
        data = np.vstack((np.zeros(8), data))
        data = np.vstack((np.zeros(8), data))
        # 小波变换
    else:
        data = pywt.wavedec(data, wavelet='db3', level=3)
        data = data[0]
        data = pywt.wavedec(data, wavelet='db2', level=2)[0]
        data = np.vstack((np.zeros(8), data.T))

    # 转换为 时序-通道 追加一个零点在转换回 通道-时序
    data = pywt.threshold(data, 15, 'hard')  # 再次阈值滤波
    data = np.abs(data)  # 反转
    return data  # 转换为 时序-通道 便于rnn输入

def emg_wave_trans(data_set):
    res_list = []
    for each_cap in data_set:
        cap = wavelet_trans(each_cap)
        res_list.append(cap)
    return res_list

def eliminate_zero_shift(data):
    zero_point = []
    for each_chanel in range(len(data[0])):
        count_dic = {}
        for each_cap in range(len(data)):
            if count_dic.get(data[each_cap][each_chanel]) is None:
                count_dic[data[each_cap][each_chanel]] = 1
            else:
                count_dic[data[each_cap][each_chanel]] += 1
        max_occr = 0
        value = 0
        for each_value in count_dic.keys():
            if max_occr < count_dic[each_value]:
                max_occr = count_dic[each_value]
                value = each_value
        if max_occr > 1:
            zero_point.append(value)
        else:
            zero_point.append(0)
    zero_point = np.array(zero_point)
    data -= zero_point
    return data

def expand_emg_data(data):
    expnded = []
    for each_data in data:
        each_data_expand = expand_emg_data_single(each_data)
        expnded.append(np.array(each_data_expand))
    return expnded

def expand_emg_data_single(data):
    expanded_data = None
    for each_dot in range(len(data)):
        if each_dot % 2 == 0:
            continue  # 只对偶数点进行左右扩展
        if each_dot - 1 < 0:
            left_val = data[each_dot]
        else:
            left_val = data[each_dot - 1]

        if each_dot + 1 >= len(data):
            right_val = data[each_dot]
        else:
            right_val = data[each_dot + 1]

        center_val = data[each_dot]
        x = np.arange(0, 2, 1)
        y = np.array([left_val, center_val])
        left_line_args = np.polyfit(x, y, 1)
        y = np.array([center_val, right_val])
        right_line_args = np.polyfit(x, y, 1)

        dot_expanded_data = None
        for each_channel in range(8):
            each_channel_dot_expanded = None

            poly_left = np.poly1d(left_line_args[:, each_channel])
            expand_range = []
            for i in range(8):
                expand_range.append(0.125 * i)

            for dot in expand_range:
                if each_channel_dot_expanded is None:
                    each_channel_dot_expanded = np.array(poly_left(dot))
                else:
                    each_channel_dot_expanded = np.vstack((each_channel_dot_expanded, poly_left(dot)))

            poly_right = np.poly1d(right_line_args[:, each_channel])
            for dot in expand_range:
                if each_channel_dot_expanded is None:
                    each_channel_dot_expanded = np.array(poly_right(dot))
                else:
                    each_channel_dot_expanded = np.vstack((each_channel_dot_expanded, poly_right(dot)))

            if dot_expanded_data is None:
                dot_expanded_data = each_channel_dot_expanded
            else:
                dot_expanded_data = np.hstack((dot_expanded_data, each_channel_dot_expanded))

        if expanded_data is None:
            expanded_data = dot_expanded_data
        else:
            expanded_data = np.vstack((expanded_data, dot_expanded_data))


    return expanded_data

# data scaling

"""
maxmin scale = (val - min) / (max - min) 
即数据在最大值最小值直接的比例
scale值阈值的设置是根据 scikit MinMax的的处理方法
scale数组中实际存储的是最大值减最小值的倒数  值越大 说明数据波动越小
如果scale时最大最小值相差很小 则不进行min max 的缩放scale 避免放大噪声
min 数组中存的是最小值 乘以scale 数组的值 相当于数据基准偏移量
数据一般都有一个小偏移量 所以数据最好都进行一下偏移修正
在不进行scale时 偏移量应还原成数据自身的偏移量 所以做之前乘法的逆运算 获取原始偏移量
"""

class DataScaler:
    """
    全局归一化scaler
    每次在生成训练数据时 根据所有数据生成一个这样的全局scaler
    在特征提取完成后 使用其进行scaling
    目前有的类型：

    'cnn',
        'cnn_acc',
        'cnn_gyr',
        'cnn_emg',
    """

    def __init__(self, scale_data_path):
        """
        :param scale_data_path: 放有scale数据文件的路径 加载scale向量
        """
        self.scale_data_path = os.path.join(scale_data_path, 'scale_data')
        self.scaler = {
            'minmax': preprocessing.MinMaxScaler(),
            # 'robust': preprocessing.RobustScaler()
        }
        self.scale_datas = {}
        try:
            file_ = open(self.scale_data_path, 'rb')
            self.scale_datas = pickle.load(file_)
            file_.close()
            print("curr scalers' type: \n\"%s\"" % str(self.scale_datas.keys()))
        except FileNotFoundError:
            print("cant load scale data, please generated before use")
            return

    def normalize(self, data, scale_type, data_type=None):
        """
        start normalize
        :param data: input
        :param scale_type: the scale method
        :param data_type:
        :return:
        """
        # 在元组中保存scale使用的min 和scale数据
        if data_type is not None:
            type_name = scale_type + '_' + data_type
        else:
            type_name = scale_type + '_all'

        if scale_type == 'minmax':
            self.scaler[scale_type].min_ = self.scale_datas[type_name][0]
            self.scaler[scale_type].scale_ = self.scale_datas[type_name][1]
            data = self.scaler[scale_type].transform(data)
            data = np.where(data < 0, 0, data)
            return np.log(1.7*data+1)
        elif scale_type == 'robust':
            self.scaler[scale_type].center_ = self.scale_datas[type_name][0]
            self.scaler[scale_type].scale_ = self.scale_datas[type_name][1]
            return self.scaler[scale_type].transform(data)


    def generate_scale_data(self, data, scale_type, data_type):
        """
        根据全局的数据生成scale vector
        :param data: 全局数据
        :param scale_type: 归一化方式 e.g. MinMax
        :param data_type: 数据类型  acc emg gyr all
        :return:
        """
        scale_data_name = '%s_%s' % (scale_type, data_type)
        if scale_type == 'minmax':
            data_range = 1.0
            max_ = np.percentile(data, 99.995, axis=0)
            min_ = np.percentile(data, 0.005, axis=0)
            min_ = np.where(abs(min_) < 0.00000001, 0, min_)

            print('max: \n' + str(max_))
            print('min: \n' + str(min_))
            scale_ = data_range / _handle_zeros_in_scale(max_ - min_)
            min_ = 0 - min_ * scale_
            self.scale_datas[scale_data_name] = (min_, scale_)


    def split_scale_vector(self, scale_name, vector_names, vector_range):
        """
        拆分scale vactor  生成是将模型各个特征输入拼接到一起生成的vector
        为了便于使用， 将不同特征的数据拆开
        :param scale_name: 待拆分的scale
        :param vector_names: 拆分后各个scale 的名字
        :param vector_range: 各个子scale对于原scale的范围
        """
        if len(vector_names) != len(vector_range):
            raise ValueError("names and ranges doesn't match")
        target_scale = self.scale_datas[scale_name]
        min_ = target_scale[0]
        scale_ = target_scale[1]
        for each in range(len(vector_names)):
            scale_data_name = '%s_%s' % (scale_name, vector_names[each])
            range_ = vector_range[each]
            self.scale_datas[scale_data_name] = (min_[range_[0]: range_[1]],
                                                    scale_[range_[0]: range_[1]])

    def store_scale_data(self):
        """
        将各个scale保存至文件
        """
        file_ = open(self.scale_data_path, 'wb')
        pickle.dump(self.scale_datas, file_, protocol=2)
        file_.close()

    def __str__(self):
        return "curr scalers' type: \n\"%s\"" % str(self.scale_datas.keys())


def _handle_zeros_in_scale(scale, copy=True):
    ''' Makes sure that whenever scale is zero, we handle it correctly.

    This happens in most scalers when we have constant features.'''

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale


