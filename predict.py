import os
import numpy as np
import torch

from options import Options
from model import Model
from torchvision.transforms import functional as F
from PIL import Image
import yaml

def save_to_local(tensor_list, folder):
    for idx_, tensor in enumerate(tensor_list):
        img = F.to_pil_image(tensor.squeeze())
        img.save(os.path.join(folder, "predict_%s.jpg" % idx_))


def predict(net, data, save_path=None):
    images, actions = data
    # images = [F.to_tensor(F.resize(F.to_pil_image(im), (opt.height, opt.width))).unsqueeze(0).to(opt.device)
    images = [im.unsqueeze(0).to(opt.device)
              for im in torch.from_numpy(images).unbind(0)]
    actions = [ac.unsqueeze(0).to(opt.device) for ac in torch.from_numpy(actions).unbind(0)]
    # states = [st.unsqueeze(0).to(opt.device) for st in torch.from_numpy(states).unbind(0)]
    # print(images.shape, actions.shape)
    with torch.no_grad():
        gen_images, gen_states = net(images, actions)
        save_images = images[:opt.context_frames] + gen_images[opt.context_frames-1:]
        if save_path:
            save_to_local(save_images, save_path)


if __name__ == '__main__':
    
    images = [np.array(Image.open("/media/joonho1804/Storage1/dirt/filtered_arc_seq_dynamics_data/0/source_seg.png"))]
    # print(images.shape)
    images = images.reshape((1, -1)) # emulate batchsize 1 sequence size 1
    path = "/media/joonho1804/Storage1/dirt/filtered_arc_seq_dynamics_data/0/source_particles.yaml"
    with open(path, 'r') as f:
        t = yaml.load(f, Loader=yaml.Loader)
        actions = np.concatenate([t['node_features'].flatten(), t['edge_features']])

    actions = actions#.reshape((1, 1, -1)) # emulate batchsize 1 sequence size 1
    opt = Options().parse()
    opt.pretrained_model = "/media/joonho1804/Storage1/dirt/physical_interaction_video_prediction_pytorch/experiment_all_nodes_multi/net_epoch_9.pth"
    opt.sequence_length = 2
    m = Model(opt)
    m.load_weight()
    net = m.net

    predict(net, (images, actions), save_path="./predict/")




