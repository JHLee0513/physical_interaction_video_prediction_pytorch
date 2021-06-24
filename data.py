import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import yaml

from sklearn.model_selection import train_test_split

# IMG_EXTENSIONS = ('.npy',)

def get_idx(x):
    return int(x.split(".")[0].split("_")[1])

def make_dataset(path):
    source_folders = os.path.join(path, 'source_segmentations')
    action_folders = os.path.join(path, 'source_tools')
    target_folders = os.path.join(path, 'next_segmentations')

    if os.path.exists(source_folders) + os.path.exists(action_folders) + os.path.exists(target_folders) != 3:
        raise FileExistsError('some subfolders from data set do not exists!')

    samples = []
    samples_list = os.listdir(source_folders)
    samples_list = sorted(samples_list, key = get_idx)
    # sequence_length = 6
    # num_sequences = len(samples_list) // sequence_length
    # for i in range(num_sequences):
    #     seq = []
    #     for j in range(sequence_length):
    #         sample = samples_list[i + j]
    #         image, action, target = os.path.join(source_folders, sample), os.path.join(action_folders, sample), os.path.join(target_folders, sample)
    #         seq.append((image, action, target))
    #     samples.append(seq)
    # return samples
    # A bit of weird indexing on the images
    episode_length = 30
    num_episodes = len(samples_list) // episode_length
    for i in range(num_episodes):
        seq = []
        for j in range(1,6):
            sample = samples_list[i*episode_length + j]
            image, action, target = os.path.join(source_folders, sample), os.path.join(action_folders, sample), os.path.join(target_folders, sample)
            seq.append((image, action, target))
        samples.append(seq)
        seq = []

        for j in range(6,31):
            if j % 6 == 0:
                if len(seq) == 5:
                    samples.append(seq)
                seq = []
                continue
            sample = samples_list[i*episode_length + j]
            image, action, target = os.path.join(source_folders, sample), os.path.join(action_folders, sample), os.path.join(target_folders, sample)
            seq.append((image, action, target))
    return samples

def image_loader(path):
    samples = Image.open(path)
    return samples

def yaml_loader(path):
    if path.split("/")[-1][:3] == "img":
        # filename is image format, switch to tool yaml ext
        root = "/".join(path.split("/")[:-1])
        idx = path.split("/")[-1].split("_")[1].split(".")[0] # get number
        path = os.path.join(root, "particles_" + idx + ".yaml")
    with open(path) as file:
        particles_list = yaml.load(file, Loader=yaml.FullLoader)
        speed = float(particles_list['speed'])

        tool_list = particles_list['tool']
        x = []
        y = []
        for particle in tool_list:
            x.append(particle[0])
            y.append(particle[1])
            hx, hy = particle[2], particle[3]
        x = np.array(x).mean()
        y = np.array(y).mean()

        heading = np.arctan2(hx, hy)

    return np.array([x, y, speed, heading])


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
        images = []
        targets = []
        actions = []
        for (image, action, target) in samples_list:
        # image, action, target = self.samples[index]

            image = image_loader(image)
            target = image_loader(target)
            action = yaml_loader(action)
            action = torch.tensor(action)

            image = self.image_transform(image)
            target = self.image_transform(target)

            images.append(image.unsqueeze(0))
            targets.append(target.unsqueeze(0))
            actions.append(action.unsqueeze(0))

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
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])

    samples = make_dataset(opt.data_dir)

    # print(samples)

    train, val = train_test_split(samples, test_size = 0.2, random_state = 42)


    # print(len(samples))

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

