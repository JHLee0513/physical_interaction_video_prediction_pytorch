import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import yaml
from glob import glob
from tqdm import tqdm

from sklearn.model_selection import train_test_split

# IMG_EXTENSIONS = ('.npy',)

def get_idx(x):
    return int(x.split(".")[0].split("_")[1])

def make_dataset(path, sequence_length = 30):
    """
    New sampling method. From each subseqence available sequence is subsampled.
    """

    samples = []
    print("Loading dataset...")
    for i, folder in enumerate(tqdm(os.listdir(path))):
        # if i > 50:
        #     break
        folder = os.path.join(path, folder)
        source_img = os.path.join(folder, 'source_seg.png')
        next_tool = os.path.join(folder, 'next_tool')
        next_seg = os.path.join(folder, 'next_seg')

        source_particle_fn = os.path.join(folder, 'source_particles.yaml')
        # get tool features, for now concatenate everything together
        tool_features = [source_particle_fn] + sorted(glob(next_tool + "/*"))
        # for tool_file in [source_particle_fn] + sorted(glob(next_tool + "/*")):
        #     with open(tool_file, 'r') as f:
        #         t = yaml.load(f, Loader=yaml.Loader)
        #         source_tool = np.concatenate([t['node_features'].flatten(), t['edge_features']])
        #         tool_features.append(source_tool)
        
        images = [source_img] + sorted(glob(next_seg + "/*.png"))

        # print(len(tool_features), len(images))
        items = []
        for a in zip(images[:-1], tool_features, images[1:]):
            items.append(list(a))
        samples.append(items)
            # samples.append(list(a))

        # samples += [list(a) for a in zip(images[:-1], tool_features, images[1:])]
        # print(len([list(a) for a in zip(images[:-1], tool_features, images[1:])][0]))
        # print(list(zip(images[:-1], tool_features, images[1:])))


        # tool_node_features = []
        # action_edge_features = []
        # next_tools = sorted(glob.glob(next_tool))
        # for tool_file in next_tools:
        #     with open(tool_file, 'r') as f:
        #         t = yaml.load(f, Loader=yaml.Loader)

        #     tool_node_features.append(
        #         torch.from_numpy(np.asarray(t['node_features'])).float())
        #     edge_features = torch.from_numpy(np.asarray(t['edge_features'])).float()
        #     action_edge_features.append(edge_features)



        # print(np.fromfile(particles_list['edge_features']))
    #     speed = float(particles_list['speed'])

    #     tool_list = particles_list['tool']
    #     x = []
    #     y = []
    #     for particle in tool_list:
    #         x.append(particle[0])
    #         y.append(particle[1])
    #         hx, hy = particle[2], particle[3]
    #     x = np.array(x).mean()
    #     y = np.array(y).mean()

    #     heading = np.arctan2(hx, hy)

    # return np.array([x, y, speed, heading])

        # for i in range()

        # source_folder = os.path.join(folder, 'source_seg')
        # action_folder = os.path.join(folder, 'source_tool')
        # target_folder = os.path.join(folder, 'next_seg')

        # if os.path.exists(source_folder) + os.path.exists(action_folder) + os.path.exists(target_folder) != 3:
        #     raise FileExistsError('some subfolders from data set do not exist!')

    # samples_list = glob(source_folder + "/*.png")
    # num_episodes = len(samples_list) // sequence_length
    # for i in range(num_episodes):
    #     seq = []
    #     for j in range(sequence_length):
    #         sample = samples_list[i*sequence_length + j]
    #         # print(sample)
    #         image, action, target = os.path.join(source_folder, sample), os.path.join(action_folder, sample), os.path.join(target_folder, sample)
    #         seq.append((image, action, target))
    #     samples.append(seq)
    return samples

def image_loader(path):
    samples = Image.open(path)
    return samples

def yaml_loader(path):
    # print(path.split("/")[-1][:3])
    # if path.split("/")[-1][:3] == "img":
    #     # filename is image format, switch to tool yaml ext
    #     root = "/".join(path.split("/")[:-1])
    #     idx = path.split("/")[-1].split("_")[1].split(".")[0] # get number
    #     path = os.path.join(root.replace("source_seg", "source_tool"), "particles_" + idx + ".yaml")
    #     # print(path)

    # with open(path) as file:
    #     particles_list = yaml.load(file, Loader=yaml.FullLoader)
    #     speed = float(particles_list['speed'])

    #     tool_list = particles_list['tool']
    #     x = []
    #     y = []
    #     for particle in tool_list:
    #         x.append(particle[0])
    #         y.append(particle[1])
    #         hx, hy = particle[2], particle[3]
    #     x = np.array(x).mean()
    #     y = np.array(y).mean()

    #     heading = np.arctan2(hx, hy)

    # return np.array([x, y, speed, heading])
    ### new yaml loaderf, handles all nodes
    with open(path, 'r') as f:
        t = yaml.load(f, Loader=yaml.Loader)
        source_tool = np.concatenate([t['node_features'].flatten(), t['edge_features']])
        # tool_features.append(source_tool)
        return source_tool


class PushDataset(Dataset):
    def __init__(self, samples, image_transform=None, device='cpu', size=(128,128)):
        # if not os.path.exists(root):
        #     raise FileExistsError('{0} does not exists!'.format(root))
        self.image_transform = image_transform
        # self.samples = make_dataset(root)
        self.samples = samples
        self.device = device

    def __getitem__(self, index):
        samples_list = self.samples[index]
        # print(samples_list)
        images = []
        targets = []
        actions = []
        for (image, action, target) in samples_list:
            # print(image, action, target)
        # image, action, target = self.samples[index]

            image = image_loader(image)
            target = image_loader(target)
            action = yaml_loader(action)
            action = torch.tensor(action)

            image = self.image_transform(image)
            target = self.image_transform(target)

            # print(action.shape, image.shape, target.shape)

            images.append(image.unsqueeze(0))
            targets.append(target.unsqueeze(0))
            actions.append(action.unsqueeze(0))
        # print(images)
        images = torch.cat(images)
        targets = torch.cat(targets)
        actions = torch.cat(actions)

        # print(images.shape, targets.shape, actions.shape)

        return images.to(self.device).float(), actions.to(self.device).float(), targets.to(self.device).float()

    def __len__(self):
        return len(self.samples)

def build_dataloader(opt):

    tf = transforms.Compose([
        # transforms.Resize((64,64)),
        transforms.Resize((128,128)),
        transforms.ToTensor(),
    ])

    samples = make_dataset(opt.data_dir)
    # print(samples)


    train, val = train_test_split(samples, test_size = 0.2, random_state = 42)


    train_ds = PushDataset(
        samples = train,
        image_transform=tf,
        device=opt.device
    )

    testseen_ds = PushDataset(
        samples = val,
        image_transform=tf,
        device=opt.device
    )

    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batch_size, shuffle=True, drop_last=False)
    testseen_dl = DataLoader(dataset=testseen_ds, batch_size=opt.batch_size, shuffle=False, drop_last=False)
    return train_dl, testseen_dl

