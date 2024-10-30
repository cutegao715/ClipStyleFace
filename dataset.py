import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from pytorch3d.io import load_obj
import torchvision


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, device):
        self.device = device
        self.frozen_img_pth = os.listdir('F:/datasets3/train_psp_w/train_images/')
        self.latents_pth = os.listdir('F:/datasets3/train_psp_w/train_latents/')
        self.obj_pth = os.listdir('F:/datasets3/train_psp_w/train_shapes/')

    def __len__(self):
        return len(self.latents_pth)

    def __getitem__(self, idx):
        ind = self.latents_pth[idx].split('.')[0]
        frozen_img = Image.open(os.path.join('F:/datasets3/train_psp_w/train_images/', ind + '.jpg'))
        frozen_img = torchvision.transforms.ToTensor()(frozen_img).to(self.device)
        latents = torch.load(os.path.join('F:/datasets3/train_psp_w/train_latents/', ind + '.pt'))
        verts, _, _ = load_obj(os.path.join('F:/datasets3/train_psp_w/train_shapes/', ind + '_template.obj'))

        texture = Image.open(os.path.join('F:/datasets3/train_psp_w/train_texture/', ind + '.jpg'))
        texture = torchvision.transforms.ToTensor()(texture).to(self.device)
        return frozen_img, torch.from_numpy(latents).squeeze().to(self.device), verts.to(self.device), texture
