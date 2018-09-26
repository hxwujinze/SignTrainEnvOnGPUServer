from __future__ import division
import pickle
import os
import random


import numpy as np
import torch
import torch.utils.data as torch_data

from process_data_dev import DATA_DIR_PATH
from process_data import DataScaler

class MyDataset(torch_data.Dataset):
    def __init__(self, data_set):
        self.data_set = data_set

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, item):
        item = self.data_set[item]
        data_mat = torch.from_numpy(item[0]).float()
        label = torch.from_numpy(np.array(item[1], dtype=int))
        return data_mat, label

class SiameseNetworkTrainDataSet:
    """
    生成随机的相同或者不同的数据对进行训练
    data set for contrastive loss
    """

    def __init__(self, data):
        """
        spilt data set into set of each categories as dict
        :param data:
        """
        self.data_len = len(data)
        self.data = data
        self.data_dict = {}
        for each_data, each_label in data:
            each_label = int(each_label)
            if self.data_dict.get(each_label) is None:
                self.data_dict[each_label] = [each_data]
            else:
                self.data_dict[each_label].append(each_data)
        self.class_cnt = 69

    # 要保证network尽可能见过最多class 并且能对他们进行正误的分辨
    def __getitem__(self, item):
        """
        50% probability, get same data or different data
        :param item:
        :return:
        """
        x1_ = random.choice(self.data)
        x1_label = x1_[1]
        x1_ = x1_[0]
        get_same = True if random.random() > 0.5 else False
        if get_same:
            x2_ = random.choice(self.data_dict[x1_label])
        else:
            x2_label = x1_label
            while x2_label == x1_label :
                x2_label = random.randint(1, self.class_cnt)
                while self.data_dict.get(x2_label) is None:
                    x2_label = random.randint(1, self.class_cnt)
            x2_ = random.choice(self.data_dict[x2_label])

        return (torch.from_numpy(x1_).float(),torch.from_numpy(x2_).float()), \
               0.0 if get_same else 1.0



    def look_input_data(self):
        """
        for test
        输出几个训练数据看看
        :return:
        """
        import matplotlib.pyplot as plt
        for i in range(20):
            each_sample = self[i]
            fig = plt.figure()
            fig.add_subplot(111, title=str(each_sample[1]))
            data_mat = each_sample[0]
            plt.plot(range(len(data_mat[0][0])), data_mat[0][0])
            plt.plot(range(len(data_mat[1][0])), data_mat[1][0])
        plt.show()


    def __len__(self):
        return self.data_len*6


class TripletLossDataSet(torch.utils.data.Dataset):
    """
    data set for triplet loss
    输出  categories x batch_k 个数据
    每个category的数据集中在一起
    """
    def __len__(self):
        return self.data_len // self.batch_size

    def __getitem__(self, index):
        """
        iter the batch by the index
        every getitem method return a batch
        :param index:
        :return:
        """
        data = []
        labels = []
        for each_cate in range(self.class_cnt):
            try:
                data.extend(random.sample(self.data_dict[each_cate], self.batch_k))
                labels.extend([each_cate for _ in range(self.batch_k)])
            except:
                continue
            # print(random.sample(self.data_dict[each_cate],self.batch_k))
        data = np.array(data)
        return torch.from_numpy(data).float(), \
               torch.from_numpy(np.array(labels, dtype=int))

    def __init__(self, data, batch_k):
        """
        spilt data set into set of each categories as dict
        :param data:
        """
        torch.utils.data.Dataset.__init__(self)
        self.data_len = len(data)
        self.class_cnt = 69
        self.batch_k = batch_k
        self.batch_size = self.class_cnt * self.batch_k
        self.data = data
        self.data_dict = {}
        for each_data, each_label in data:
            each_label = int(each_label)
            if self.data_dict.get(each_label) is None:
                self.data_dict[each_label] = [each_data]
            else:
                self.data_dict[each_label].append(each_data)


class HybridModelDataset:
    def __init__(self, data_set):
        self.data_set = data_set

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, item):
        item = self.data_set[item]
        data_mat = torch.from_numpy(item[0]).float()
        emg_data_mat = torch.from_numpy(item[1]).float()
        label = torch.from_numpy(np.array(item[2], dtype=int))
        return [data_mat, emg_data_mat], label

def generate_data_set(split_ratio, data_set_type, batch_k=16):
    """
    generate the train/test data set
    split the data set according to split ratio
    and load up to DataSet obj
    :param split_ratio: how much data save as test data, or train data
    :param data_set_type: which train data set type you load up in
                          pass the class into that
    :param batch_k: if the dataset for triplet loss, need to provide batch_k
    :return: dict {
        'train': train DataSet,
        'test': test DataSet
    }
    """
    data_set_name = 'new_train_data'
    if data_set_type == HybridModelDataset:
        data_set_name += '_hybrid'
    data_path = os.path.join(DATA_DIR_PATH, data_set_name)
    with open(data_path, 'r+b') as f:
        data_set = pickle.load(f)
        
    print('data set size %d' % len(data_set))
    random.shuffle(data_set)
    for each in range(len(data_set)):
        if data_set_type == HybridModelDataset:
            data_set[each] = (data_set[each][0].T,
                              data_set[each][1].T,
                              data_set[each][2])
        else:
            data_set[each] = (data_set[each][0].T,
                              data_set[each][1])

    train_data = data_set[int(len(data_set)*split_ratio) :]
    test_data = data_set[: int(len(data_set)*split_ratio)]

    if data_set_type is not TripletLossDataSet:
        return {
            'train': data_set_type(train_data),
            'test': data_set_type(test_data)
        }
    else:
        return {
            'train': data_set_type(train_data, batch_k),
            'test': SiameseNetworkTrainDataSet(test_data)
        }

