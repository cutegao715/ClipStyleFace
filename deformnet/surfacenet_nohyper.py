import torch
from torch import nn
import deformnet.modules as modules


class SurfaceDeformationField(nn.Module):
    def __init__(self, num_instances, latent_dim=128, model_type='sine', hyper_hidden_layers=1, num_hidden_layers=3, hyper_hidden_features=256,hidden_num=128, **kwargs):
        super().__init__()

        self.latent_dim = latent_dim
        self.deform_net=modules.SingleBVPNet(type=model_type,mode='mlp', hidden_features=hidden_num, num_hidden_layers=num_hidden_layers, in_features=3,out_features=3)

    # for training
    def forward(self, coords):
        model_output = self.deform_net(coords)
        displacement = model_output['model_out']
        return displacement
