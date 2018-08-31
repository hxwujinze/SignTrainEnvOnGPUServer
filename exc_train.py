import os

import models.CNN_model as CNN_model
import models.verify_model as SN
import torch

if __name__ == "__main__":

    m = SN.SiameseNetwork()
    dirs = os.listdir('.')
    for each in dirs:
        if each.startswith('verify') and each.endswith('.pkl'):
            m.load_state_dict(torch.load(each))
            print('load params %s' % each)
            break
    m.exc_train()
    m = CNN_model.CNN()
    m.exc_train()
