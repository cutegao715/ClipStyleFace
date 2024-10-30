import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from pytorch3d.io import load_obj
import torchvision


class ValDataset(torch.utils.data.Dataset):
    def __init__(self, device, style):
        self.device = device
        self.latents_pth = os.listdir('./datasets/latents/')
        self.style = style

    def __len__(self):
        return len(self.latents_pth)

    def __getitem__(self, idx):
        ind = self.latents_pth[idx].split('.')[0]

        frozen_img = Image.open(os.path.join('./datasets/images/', ind + '.jpg'))
        frozen_img = torchvision.transforms.ToTensor()(frozen_img).to(self.device)
        neg_img = Image.open(os.path.join('./results_cin/', self.style, '(\'{}\',)'
                                          .format(ind), '4', '10th_view.jpg'))
        neg_img = torchvision.transforms.ToTensor()(neg_img).to(self.device)
        latents = torch.load(os.path.join('./datasets/latents/', ind + '.pth'))
        verts, _, _ = load_obj(os.path.join('./datasets/shapes/', ind + '_template.obj'))

        texture = Image.open(os.path.join('./datasets/texture/', ind + '.png'))
        texture = torchvision.transforms.ToTensor()(texture).to(self.device)
        return frozen_img, latents.squeeze().to(self.device), verts.to(self.device), texture, ind, neg_img
