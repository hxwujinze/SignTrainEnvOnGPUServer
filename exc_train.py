import os

import algorithm_models.CNN_model as CNN_model
import algorithm_models.verify_model as SN
import torch

if __name__ == "__main__":


    m = CNN_model.CNN()
    m.exc_train()
    m = SN.SiameseNetwork(True)
    m.exc_train()
