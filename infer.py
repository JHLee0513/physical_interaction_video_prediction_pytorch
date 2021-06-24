from options import Options
from model import Model

def train():
    opt = Options().parse()
    model = Model(opt)
    # model.load_weight("./weights/net_epoch_9.pth")
    model.inference("./viz")


if __name__ == '__main__':
    train()