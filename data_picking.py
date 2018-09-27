import json
import os
import pickle
import shutil
import random

import numpy as np
import torch
import torch.nn.functional as F

from algorithm_models.classify_model import CNN
from algorithm_models.verify_model import SiameseNetwork
import data_augmentation
import process_data_dev
from process_data_dev import DATA_DIR_PATH

def picking_by_distribution(manual=False, print_plt=False):
    """
        根据的数据的分布情况计算数据质量，并根据质量筛选数据
        可以选择手动筛选以及人工筛选， 以及是否显示数据的可视化曲线图
        :param manual
        :param print_plt
        :return:
        """
    source_dir = 'resort_data'
    source_dir_abs_path = os.path.join(process_data_dev.DATA_DIR_PATH, source_dir)
    stat_book = process_data_dev.statistics_data(source_dir)
    target_dir_abs_path = os.path.join(process_data_dev.DATA_DIR_PATH, 'cleaned_data_test')
    if not os.path.exists(target_dir_abs_path):
        os.makedirs(target_dir_abs_path)

    scanned_book = {}
    scanned_book_f = 'scanned_data.dat'
    # 记录已经检查过的数据 避免重复检查
    if os.path.exists(scanned_book_f):
        f = open(scanned_book_f, 'r+b')
        scanned_book = pickle.load(f)

    print(json.dumps(scanned_book, indent=2))

    for each_sign in [28]:  # , 24, 27 , 31, 34]:
        for each_batch in stat_book[each_sign]['occ_pos']:
            try:
                scanned_book[each_sign].index(each_batch)
                continue
            except (ValueError, KeyError):
                pass

            each_batch = each_batch.split(' ')
            date = each_batch[0]
            batch_id = each_batch[1]

            distribution = data_augmentation.get_distribution_single((int(batch_id),
                                                                      date,
                                                                      int(each_sign)))

            if distribution[-1] is not None:
                judge_res = distribution[-1]['judge_res']
                print(judge_res)
            else:
                judge_res = False

            print("show data sign %s %s %s" % (each_sign, date, batch_id))

            if print_plt:
                process_data_dev.print_train_data(sign_id=int(each_sign),
                                                  batch_num=int(batch_id),
                                                  data_cap_type='acc',
                                                  data_feat_type='poly_fit',
                                                  capture_date=date,
                                                  data_path=source_dir,
                                                  for_cnn=True)

                process_data_dev.print_train_data(sign_id=int(each_sign),
                                                  batch_num=int(batch_id),
                                                  data_cap_type='gyr',
                                                  data_feat_type='poly_fit',
                                                  capture_date=date,
                                                  data_path=source_dir,
                                                  for_cnn=True)
            mark = '%s %s' % (date, batch_id)
            if scanned_book.get(each_sign) is None:
                scanned_book[each_sign] = [mark]
            else:
                scanned_book[each_sign].append(mark)

            if manual:
                print("save it? y/n")
                res = input()
                if res == 'y':
                    judge_res = True

            if judge_res:
                for each_type in ['Acceleration', 'Emg', 'Gyroscope']:
                    source_file_path = os.path.join(date, str(batch_id), each_type)

                    old_path = os.path.join(source_dir_abs_path, source_file_path, str(each_sign) + '.txt')
                    target_path = os.path.join(target_dir_abs_path, source_file_path)

                    if not os.path.exists(target_path):
                        os.makedirs(target_path)
                    new_path = os.path.join(target_path, str(each_sign) + '.txt')
                    if os.path.exists(new_path):
                        print("%s %s %s" % (date, batch_id, each_sign))
                    shutil.copyfile(old_path, new_path)

            with open(scanned_book_f, 'w+b') as f:
                pickle.dump(scanned_book, f)

def picking_by_verify_model():
    verify_m = SiameseNetwork(train=False)
    load_model_param(verify_m, 'verify')
    reference_vectors = os.path.join(DATA_DIR_PATH, 'reference_verify_vector')


    with open(reference_vectors, 'rb') as f:
        reference_vectors = pickle.load(f)
    with open(os.path.join(DATA_DIR_PATH, 'new_train_data'), 'rb') as f:
        all_data = pickle.load(f)

    random.shuffle(all_data)
    print("init_data len %d" % len(all_data))
    vaild_data = []
    cnter_book = {}
    range_iter = 0
    batch_size = 1000
    while range_iter < len(all_data):
        if range_iter % batch_size == 0:
            print('progress %d / %d' %(range_iter, len(all_data)))
        seg_mat = all_data[range_iter: range_iter + batch_size]
        range_iter += batch_size
        input_seg = []
        for each_mat in seg_mat:
            if cnter_book.get(each_mat[1]) is None:
                cnter_book[each_mat[1]] = {
                    'valid':0,
                    'all':0,
                }
            cnter_book[each_mat[1]]['all'] += 1
            input_seg.append(each_mat[0].T)
        x = torch.from_numpy(np.array(input_seg))
        x = x.double()
        feat_vec = verify_m(x)
        for each in range(len(input_seg)):
            try:
                refer_vector = reference_vectors[seg_mat[each][1]][0]
                threshold = reference_vectors[seg_mat[each][1]][1]
            except:
                continue
            torch_vec = torch.from_numpy(feat_vec[each].detach().numpy())
            refer_vector = torch.from_numpy(refer_vector.detach().numpy())
            dis = torch.sqrt(torch.sum((torch_vec-refer_vector)**2))
            # dis = F.pairwise_distance(torch_vec, refer_vector)
            # print(dis)
            threshold += 0.065
            # print("%f -> %f" % (dis, threshold))
            if dis < threshold:
                vaild_data.append(seg_mat[each])
                cnter_book[seg_mat[each][1]]['valid'] += 1
        # if range_iter % 4000 == 0:
        #     for each_sign in cnter_book.keys():
        #         print('sign %d: %d / %d' % (each_sign,
        #                                     cnter_book[each_sign]['valid'],
        #                                     cnter_book[each_sign]['all']))
        #     print('remain data %d / %d' % (len(vaild_data), len(all_data)))
    print('pick process done')

    for each_sign in cnter_book.keys():
        print('sign %d: %d / %d' % (each_sign,
                                    cnter_book[each_sign]['valid'],
                                    cnter_book[each_sign]['all']))
    print('remain data %d / %d' % (len(vaild_data), len(all_data)))

    with open(os.path.join(DATA_DIR_PATH, 'new_train_data_picked'), 'wb') as f:
        pickle.dump(vaild_data, f)


def load_model_param(model, model_name):
    for root, dirs, files in os.walk(DATA_DIR_PATH):
        for file_ in files:
            file_name_split = os.path.splitext(file_)
            if file_name_split[1] == '.pkl' and file_name_split[0].startswith(model_name):
                print('load model params %s' % file_name_split[0])
                file_ = os.path.join(DATA_DIR_PATH, file_)
                model.load_state_dict(torch.load(file_))
                model.double()
                model.eval()
                return model


def main():
    # picking_by_distribution(manual=False)
    picking_by_verify_model()

if __name__ == '__main__':
    main()
    # process_data_dev.statistics_data('cleaned_data_test')
