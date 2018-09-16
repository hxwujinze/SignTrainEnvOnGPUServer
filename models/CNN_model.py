import  os
import math

import torch
import torch.nn as nn
import torch.utils.data.dataloader as DataLoader

# from models.make_resnet import my_resnet
# from models.make_VGG import make_vgg


LEARNING_RATE = 0.0001
EPOCH = 100
BATCH_SIZE = 128
WEIGHT_DECAY = 0.000005

class CNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        # self.convs = my_resnet(layers=[2 ,2], layer_planes=[64, 128])
        self.convs = make_vgg(input_chnl=14, layers=[2, 3], layers_chnl=[64, 128])


        self.out = nn.Sequential(
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(256, 69),

        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] *  m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        """
        just 2 conv and 3 layers FC
        makes better performance
        :param x:
        :param encode_mode: set True if just need the feature vector
        :return:
        """
        x = self.convs(x)
        x = self.out(x)
        return x


    def exc_train(self):
        # only import train staff in training env
        from train_util.data_set import generate_data_set, MyDataset
        from train_util.common_train import train
        print("CNN classify model start training")
        print(str(self))

        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        loss_func = nn.CrossEntropyLoss()
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        data_set =generate_data_set(0.06, MyDataset)
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
              save_dir='./params',
              data_set=data_set,
              data_loader=data_loader,
              test_result_output_func=test_result_output,
              cuda_mode = 1,
              print_inter=2,
              val_inter=30,
              scheduler_step_inter=50
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

def get_max_index(tensor):
    # print('置信度')
    tensor = F.softmax(tensor, dim=1)
    tensor = torch.max(tensor, dim=1)[1]
    # 对矩阵延一个固定方向取最大值
    return torch.squeeze(tensor).data.int()


def output_len(Lin, padding, kernel_size, stride):
    Lout = (Lin + 2 * padding - (kernel_size - 1) - 1) / stride + 1
    return Lout



# --------------make VGG--------------
# to avoid package import process. move all code into one file

class VGGBlock(nn.Module):
    def __init__(self, input_chnl, output_chnl, layers):
        super(VGGBlock, self).__init__()
        layers_to_add = []
        for each in range(layers):

            if each == 0:
                tmp_in = input_chnl
            else:
                tmp_in = output_chnl

            layer = nn.Sequential(
                nn.LeakyReLU(),
                nn.Conv1d(
                    in_channels=tmp_in,
                    out_channels=output_chnl,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.BatchNorm1d(output_chnl)
            )
            layers_to_add.append(layer)

            if each == layers-1:
                layers_to_add.append(nn.MaxPool1d(
                                        kernel_size=3,
                                        stride=2,
                                    ))

        self.block = nn.Sequential(
            *layers_to_add
        )


    def forward(self, x):
        x = self.block(x)
        return x


class VGGNet(nn.Module):
    def __init__(self, layers, layer_chnl, input_plane):
        """
        生成VGGNet
        :param block: 用什么样的block？
        :param layers: block里面放几层？
        :param layer_chnl: the output channel in each block
        """

        if len(layers) != len(layer_chnl):
            raise Exception('the length of layers cnt args and planes args should same')
        super(VGGNet, self).__init__()


        block_to_add = []
        for each in range(len(layers)):
            if each == 0:
                input_chnl = input_plane
            else:
                input_chnl = layer_chnl[each-1]
            block_to_add.append(self.__make_layer(input_chnl, layer_chnl[each], layers[each]))
        self.blocks = nn.Sequential(*block_to_add)
        self.out = nn.AdaptiveAvgPool1d(2)


    def forward(self, x):
        x = self.blocks(x)
        x = self.out(x)
        x =  x.view(x.size(0), -1)
        return x


    @staticmethod
    def __make_layer(input_plane, output_plane, layers):
        return VGGBlock(input_chnl=input_plane, output_chnl=output_plane, layers=layers)



def make_vgg(input_chnl, layers, layers_chnl):
    return VGGNet(input_plane=input_chnl, layers=layers, layer_chnl=layers_chnl)
