import os

import algorithm_models.classify_model as cm
import algorithm_models.verify_model as SN
import torch

if __name__ == "__main__":

    m = cm.HybridModel()
    # m = CNN_model.CNN()
    m.exc_train()
    # m = SN.SiameseNetwork(True)
    # m.exc_train()
