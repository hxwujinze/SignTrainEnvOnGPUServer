import os

import models.CNN_model as CNN_model
import models.verify_model as SN
import torch

if __name__ == "__main__":


    m = CNN_model.CNN()
    m.exc_train()
    m = SN.SiameseNetwork()
    m.exc_train()
