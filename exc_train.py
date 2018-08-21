import models.CNN_model as CNN_model
import models.verify_model as SN


if __name__ == "__main__":
    m = SN.SiameseNetwork('.', train=True)
    #m = CNN_model.CNN()
    m.exc_train()
