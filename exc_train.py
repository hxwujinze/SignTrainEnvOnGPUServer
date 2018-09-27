import os

import algorithm_models.classify_model as cm
import algorithm_models.verify_model as SN
import torch

if __name__ == "__main__":

    # m = cm.HybridClassifyModel()
    m = cm.CNN()
    m.exc_train()
    m = SN.SiameseNetwork(True)
    m.exc_train()
