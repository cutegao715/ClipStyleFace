import os
import yaml
import trimesh
import torch

import matplotlib.cm
import torch_geometric.transforms

import networkx as nx
import numpy as np
from collections import Counter
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian


def load_laplacian(verts, faces):
    mesh_verts = torch.tensor(verts, dtype=torch.float,
                              requires_grad=False)
    face = torch.tensor(faces).t().to(torch.long).contiguous()
    data = Data(pos=mesh_verts, face=face)
    data = torch_geometric.transforms.FaceToEdge(False)(data)
    data.laplacian = torch.sparse_coo_tensor(
        *get_laplacian(data.edge_index, normalization='rw'))
    return data


def batch_mm(sparse, matrix_batch):
    """
    :param sparse: Sparse matrix, size (m, n).
    :param matrix_batch: Batched dense matrices, size (b, n, k).
    :return: The batched matrix-matrix product, size (b, m, k).
    """
    batch_size = matrix_batch.shape[0]
    # Stack the vector batch into columns (b, n, k) -> (n, b, k) -> (n, b*k)
    matrix = matrix_batch.transpose(0, 1).reshape(sparse.shape[1], -1)

    # And then reverse the reshaping.
    return sparse.mm(matrix).reshape(sparse.shape[0],
                                     batch_size, -1).transpose(1, 0)


def compute_laplacian_regularizer(pred, template):
    bs, n_vert, _ = pred.shape
    laplacian = template.laplacian.to(pred.device)
    pred_laplacian = batch_mm(laplacian, pred)
    loss = pred_laplacian.norm(dim=-1) / n_vert
    return loss.sum() / bs

import torch
from torch import nn
import torch.nn.functional as F
"""
Adapted from Nerfies by Yushi
"""
# @staticmethod
def calculate_deformation_gradient(points, offset):
    # import pdb; pdb.set_trace()
    u = offset[..., 0]
    v = offset[..., 1]
    w = offset[..., 2]

    grad_outputs = torch.ones_like(u)
    grad_u = torch.autograd.grad(u, [points],
                                 grad_outputs=torch.ones_like(grad_outputs),
                                 create_graph=True)[0]
    grad_v = torch.autograd.grad(v, [points],
                                 grad_outputs=torch.ones_like(grad_outputs),
                                 create_graph=True)[0]
    grad_w = torch.autograd.grad(w, [points],
                                 grad_outputs=torch.ones_like(grad_outputs),
                                 create_graph=True)[0]

    grad_deform = torch.stack([grad_u, grad_v, grad_w], -1)  #

    return grad_deform

# loss 部分
class JacobianSmoothness(nn.Module):
    # Directly Panalize the grad of D_field Jacobian.
    def __init__(self, margin=0.5):
        super().__init__()
        # self.gradient_panelty = torch.nn.MSELoss()
        self.margin = margin

    def forward(self, gradient: torch.Tensor):
        """eikonal loss to encourage smoothness
        Args:
            gradient (torch.Tensor): B N 3?
        Returns:
            torch.Tensor: max(||gradient.norm()||_2^{2}-margin, 0)
        """
        # ?
        # import ipdb
        # ipdb.set_trace()
        # return self.gradient_panelty(torch.linalg.norm(gradient, dim=1), 1)
        grad_norm = torch.linalg.norm(gradient, dim=1).square()  # B 3 3
        return F.relu(grad_norm - self.margin).mean()