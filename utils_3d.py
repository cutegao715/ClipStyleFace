import torch
import numpy as np

from decalib.deca2 import DECA
from decalib.utils import lossfunc
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.renderer import SRenderY, set_rasterizer

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes


# cam utils
def random_cam(verts, num_views, center_elev=0., center_azim=0., std=8):
    elev = torch.cat((torch.tensor([center_elev]), torch.randn(num_views - 1) * np.pi / std + center_elev))
    azim = torch.cat((torch.tensor([center_azim]), torch.randn(num_views - 1) * 2 * np.pi / std + center_azim))

    trans_verts = []
    verts = torch.nn.functional.pad(verts, (0, 1), mode='constant', value=1)
    for i in range(num_views):
        camera_transform = get_camera_from_view(elev[i], azim[i], r=2).to(args.device)
        trans_vert = (verts @ camera_transform)
        trans_verts.append(trans_vert[0])
    trans_verts = torch.stack(trans_verts)
    return trans_verts


def generate_transformation_matrix(camera_position, look_at, camera_up_direction):
    z_axis = (camera_position - look_at)
    z_axis /= z_axis.norm(dim=1, keepdim=True)
    # torch.cross don't support broadcast
    # (https://github.com/pytorch/pytorch/issues/39656)
    if camera_up_direction.shape[0] < z_axis.shape[0]:
        camera_up_direction = camera_up_direction.repeat(z_axis.shape[0], 1)
    elif z_axis.shape[0] < camera_up_direction.shape[0]:
        z_axis = z_axis.repeat(camera_up_direction.shape[0], 1)
    x_axis = torch.cross(camera_up_direction, z_axis, dim=1)
    x_axis /= x_axis.norm(dim=1, keepdim=True)
    y_axis = torch.cross(z_axis, x_axis, dim=1)
    rot_part = torch.stack([x_axis, y_axis, z_axis], dim=2)
    trans_part = (-camera_position.unsqueeze(1) @ rot_part)
    return torch.cat([rot_part, trans_part], dim=1)


def get_camera_from_view(elev, azim, r=3.0):
    x = r * torch.cos(azim) * torch.sin(elev)
    y = r * torch.sin(azim) * torch.sin(elev)
    z = r * torch.cos(elev)
    # print(elev,azim,x,y,z)

    pos = torch.tensor([x, y, z]).unsqueeze(0)
    look_at = -pos
    direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)

    camera_proj = generate_transformation_matrix(pos, look_at, direction)
    return camera_proj


def get_camera_from_view3(elev, azim, r=3.0):
    x = r * torch.cos(azim) * torch.sin(elev)
    y = r * torch.sin(azim) * torch.sin(elev)
    if elev == 0 and azim > 0:
        y = -0.3536
    elif elev == 0 and azim < 0:
        y = 0.3536
    z = r * torch.cos(elev)
    # print(elev,azim,x,y,z)

    pos = torch.tensor([x, y, z]).unsqueeze(0)
    look_at = -pos
    direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)

    camera_proj = generate_transformation_matrix(pos, look_at, direction)
    return camera_proj


def batch_orth_proj(X, camera):
    ''' orthgraphic projection
        X:  3d vertices, [bz, n_point, 3]
        camera: scale and translation, [bz, 3], [scale, tx, ty]
    '''
    camera = camera.clone().view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    X_trans = torch.cat([X_trans, X[:,:,2:]], 2)
    shape = X_trans.shape
    Xn = (camera[:, :, 0:1] * X_trans)
    return Xn
# verts

def random_cam(verts, num_views, center_elev=0., center_azim=0., std=8):
    # azim = torch.linspace(center[0], 2 * np.pi + center[0], num_views + 1)[
    #        :-1]  # since 0 =360 dont include last element
    # elev = torch.cat((torch.linspace(center[1], np.pi / 2 + center[1], int((num_views + 1) / 2)),
    #                   torch.linspace(center[1], -np.pi / 2 + center[1], int((num_views) / 2))))

    elev = torch.cat((torch.tensor([center_elev]), torch.randn(num_views - 1) * np.pi / std + center_elev))
    azim = torch.cat((torch.tensor([center_azim]), torch.randn(num_views - 1) * 2 * np.pi / std + center_azim))

    trans_verts = []
    verts = torch.nn.functional.pad(verts, (0, 1), mode='constant', value=1)
    for i in range(num_views):
        camera_transform = get_camera_from_view(elev[i], azim[i], r=2).cuda()
        trans_vert = (verts @ camera_transform)
        trans_verts.append(trans_vert[0])
    trans_verts = torch.stack(trans_verts)
    return trans_verts


def fix_cam(verts, elev, azim):
    verts = torch.nn.functional.pad(verts, (0, 1), mode='constant', value=1)
    camera_transform = get_camera_from_view3(elev, azim, r=2).cuda()
    trans_vert = (verts @ camera_transform)
    return trans_vert


def fix_cam_batch(verts):
    azims = torch.linspace(-np.pi / 6, np.pi / 6, 3)  # since 0 =360 dont include last element
    elevs = torch.linspace(-np.pi / 6, np.pi / 6, 3)

    trans_verts = []
    verts = torch.nn.functional.pad(verts, (0, 1), mode='constant', value=1)
    for i in range(3):
        for j in range(3):
            camera_transform = get_camera_from_view3(elevs[i], azims[j], r=2).cuda()
            trans_vert = (verts @ camera_transform)
            trans_verts.append(trans_vert[0])
    trans_verts = torch.stack(trans_verts)
    return trans_verts