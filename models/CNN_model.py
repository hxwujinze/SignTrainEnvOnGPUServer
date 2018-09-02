import  os


import torch
import torch.nn as nn
import torch.utils.data.dataloader as DataLoader



LEARNING_RATE = 0.00015
EPOCH = 700
BATCH_SIZE = 64
WEIGHT_DECAY = 0.00001

class CNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        # input 14 x 64
        self.conv1 = nn.Sequential(
            # 使用VGGNet架构卷积
            nn.Conv1d(
                in_channels=14,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1,
            ),  # Lout=floor((Lin+2*padding-dilation*(kernel_size -1 ) - 1)/stride+1)
            # output 28 x 64
            nn.BatchNorm1d(64),

            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            # output 28 x 64
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=4, stride=3)  # 64 x 32
        )

        self.conv2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1
            ),  # 32 x 21
            nn.BatchNorm1d(64),

            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1
            ),  # 32 x 21
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=3, stride=2)  # 128 x 16
        )


        self.conv3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                stride=1
            ),  # 32 x 21
            nn.BatchNorm1d(128),

            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1,
                stride=1
            ),  # 32 x 21
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2, stride=1)  # 128 x 16
        )

        self.out1 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(1152, 512),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.Tanh(),  #  use tanh as activity function next to the softmax
            nn.Dropout(),
            nn.Linear(256, 69),
            nn.Softmax(),
        )



    def forward(self, x):
        """
        just 2 conv and 3 layers FC
        makes better performance
        :param x:
        :param encode_mode: set True if just need the feature vector
        :return:
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.out1(x)
        return x


    def exc_train(self):
        # only import train staff in training env
        from train_util.data_set import generate_data_set, MyDataset
        from train_util.common_train import train

        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        loss_func = nn.CrossEntropyLoss()
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        data_set =generate_data_set(0.1, MyDataset)
        data_loader = {
            'train': DataLoader.DataLoader(data_set['train'], 
                                           shuffle=True,
                                           batch_size=BATCH_SIZE,
                                           num_workers=1),
            'test': DataLoader.DataLoader(data_set['test'], 
                                          shuffle=True,
                                          batch_size=1,
                                          num_workers=1)
        }
        train(model=self,
              model_name='cnn_68',
              EPOCH=EPOCH,
              optimizer=optimizer,
              exp_lr_scheduler=lr_scheduler,
              loss_func=loss_func,
              save_dir='.',
              data_set=data_set,
              data_loader=data_loader,
              test_result_output_func=test_result_output,
              cuda_mode = 0,
              print_inter=2,
              val_inter=30,
              scheduler_step_inter=60
              )

    def load_params(self, path):
        file_list = os.listdir(path)
        target = None
        for each in file_list:
            if each.startswith("cnn_68") and each.endswith('.pkl'):
                target = each
                break
        if target is None:
            raise Exception("can't find satisfy model params")
        target = os.path.join(path, target)
        print(target)
        self.load_state_dict(torch.load(target))





def get_max_index(tensor):
    # print('置信度')
    # print(tensor.data.float()[0])
    tensor = torch.max(tensor, dim=1)[1]
    # 对矩阵延一个固定方向取最大值
    return torch.squeeze(tensor).data.int()

def test_result_output(result_list, epoch, loss):
    test_result = {}
    all_t_cnt = 0
    all_f_cnt = 0
    for each in result_list:
        target_y = each[0]
        test_output = each[1]
        if test_result.get(target_y) is None:
            test_result[target_y] = {
                't': 0,
                'f': 0
            }
        if test_output == target_y:
            all_t_cnt += 1
            test_result[target_y]['t'] += 1
        else:
            all_f_cnt += 1
            test_result[target_y]['f'] += 1
    accuracy_res = "accuracy of each sign:\n"
    for each_sign in sorted(test_result.keys()):
        t_cnt = test_result[each_sign]['t']
        f_cnt = test_result[each_sign]['f']
        accuracy_rate = t_cnt / (t_cnt + f_cnt)
        accuracy_res += "sign %d, accuracy %f (%d / %d)\n" % \
                        (each_sign, accuracy_rate, t_cnt, t_cnt + f_cnt)
    accuracy_res += "overall accuracy: %.5f\n" % (all_t_cnt / (all_f_cnt + all_t_cnt))

    print("**************************************")
    print("epoch: %s\nloss: %s\nprogress: %.2f" %
          (epoch, loss, 100 * epoch / EPOCH,))
    print(accuracy_res)
    return accuracy_res


def output_len(Lin, padding, kernel_size, stride):
    Lout = (Lin + 2 * padding - (kernel_size - 1) - 1) / stride + 1
    return Lout
