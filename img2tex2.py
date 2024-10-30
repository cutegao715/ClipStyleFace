import os
import torchvision
import torch
from tqdm import tqdm
import math
import numpy as np
import cv2


def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def get_normal(p1, p2, p3):
	return np.cross(p2-p1, p3-p1)

def mesh_angle(vertices, vertex_ids):

	normal = get_normal(np.array(vertices[vertex_ids[0]]),
								np.array(vertices[vertex_ids[1]]),
								np.array(vertices[vertex_ids[2]]))

	ang = int(angle(normal, [1,0,1])*360/math.pi)

	return ang

def tex_correction(uv_texture, angle):

    if angle < 0:
        max_pixel = 512
        arr = np.array(range(max_pixel))/max_pixel
        arr_flip = np.flip(arr, 0)
        uv_texture[200:,:max_pixel,:] = torch.flip(uv_texture, (1,))[200:,:max_pixel,:] * arr_flip[None,...,None] + uv_texture[200:,:max_pixel,:] * arr[None,...,None]
        uv_texture[200:,:max_pixel,:] = torch.flip(uv_texture, (1,))[200:,:max_pixel,:] * arr_flip[None,...,None] + uv_texture[200:,:max_pixel,:] * arr[None,...,None]
        uv_texture[200:,:max_pixel,:] = torch.flip(uv_texture, (1,))[200:,:max_pixel,:] * arr_flip[None,...,None] + uv_texture[200:,:max_pixel,:] * arr[None,...,None]
        uv_texture[200:,:max_pixel,:] = torch.flip(uv_texture, (1,))[200:,:max_pixel,:] * arr_flip[None,...,None] + uv_texture[200:,:max_pixel,:] * arr[None,...,None]

    else:
        max_pixel = -512
        arr = np.array(range(abs(max_pixel)))/abs(max_pixel)
        arr_flip = np.flip(arr, 0)
        uv_texture[200:,max_pixel:,:] = torch.flip(uv_texture, (1,))[200:,max_pixel:,:] * arr[None,...,None] + uv_texture[200:,max_pixel:,:] * arr_flip[None,...,None]
        uv_texture[200:,max_pixel:,:] = torch.flip(uv_texture, (1,))[200:,max_pixel:,:] * arr[None,...,None] + uv_texture[200:,max_pixel:,:] * arr_flip[None,...,None]
        uv_texture[200:,max_pixel:,:] = torch.flip(uv_texture, (1,))[200:,max_pixel:,:] * arr[None,...,None] + uv_texture[200:,max_pixel:,:] * arr_flip[None,...,None]
        uv_texture[200:,max_pixel:,:] = torch.flip(uv_texture, (1,))[200:,max_pixel:,:] * arr[None,...,None] + uv_texture[200:,max_pixel:,:] * arr_flip[None,...,None]

    return uv_texture

def tex_correction_eye(uv_texture, angle):

    if angle < 0:
        max_pixel = 512
        eye = 1
        arr = np.array(range(max_pixel))/max_pixel
        arr_flip = np.flip(arr, 0)
        uv_texture[:,:max_pixel,:] = torch.flip(uv_texture, (1,))[:,:max_pixel,:]
        uv_texture[:200,:200,:] = eye

    else:
        max_pixel = -512
        eye = uv_texture[:200,-200:,:].clone()
        arr = np.array(range(abs(max_pixel)))/abs(max_pixel)
        arr_flip = np.flip(arr, 0)
        uv_texture[:,max_pixel:,:] = torch.flip(uv_texture, (1,))[:,max_pixel:,:]
        uv_texture[:200,-200:,:] = eye

    return uv_texture


def tex_merge(uv_texture_r, uv_texture_c, uv_texture_l):

    max_pixel = 512
    arr = np.array(range(max_pixel))/max_pixel
    arr_flip = np.flip(arr, 0)
    uv_texture_c[200:,:max_pixel,:] = uv_texture_l[200:,:max_pixel,:] * arr_flip[None,...,None] + uv_texture_c[200:,:max_pixel,:] * arr[None,...,None]
    uv_texture_c[200:,:max_pixel,:] = uv_texture_l[200:,:max_pixel,:] * arr_flip[None,...,None] + uv_texture_c[200:,:max_pixel,:] * arr[None,...,None]
    uv_texture_c[200:,:max_pixel,:] = uv_texture_l[200:,:max_pixel,:] * arr_flip[None,...,None] + uv_texture_c[200:,:max_pixel,:] * arr[None,...,None]
    uv_texture_c[200:,:max_pixel,:] = uv_texture_l[200:,:max_pixel,:] * arr_flip[None,...,None] + uv_texture_c[200:,:max_pixel,:] * arr[None,...,None]
    uv_texture_c[200:,:max_pixel,:] = uv_texture_l[200:,:max_pixel,:] * arr_flip[None,...,None] + uv_texture_c[200:,:max_pixel,:] * arr[None,...,None]
    # uv_texture[200:,:max_pixel,:] = torch.flip(uv_texture, (1,))[200:,:max_pixel,:] * arr_flip[None,...,None] + uv_texture[200:,:max_pixel,:] * arr[None,...,None]

    max_pixel = -512
    arr = np.array(range(abs(max_pixel)))/abs(max_pixel)
    arr_flip = np.flip(arr, 0)
    uv_texture_c[200:,max_pixel:,:] = uv_texture_r[200:,max_pixel:,:] * arr[None,...,None] + uv_texture_c[200:,max_pixel:,:] * arr_flip[None,...,None]
    uv_texture_c[200:,max_pixel:,:] = uv_texture_r[200:,max_pixel:,:] * arr[None,...,None] + uv_texture_c[200:,max_pixel:,:] * arr_flip[None,...,None]
    uv_texture_c[200:,max_pixel:,:] = uv_texture_r[200:,max_pixel:,:] * arr[None,...,None] + uv_texture_c[200:,max_pixel:,:] * arr_flip[None,...,None]
    uv_texture_c[200:,max_pixel:,:] = uv_texture_r[200:,max_pixel:,:] * arr[None,...,None] + uv_texture_c[200:,max_pixel:,:] * arr_flip[None,...,None]
    uv_texture_c[200:,max_pixel:,:] = uv_texture_r[200:,max_pixel:,:] * arr[None,...,None] + uv_texture_c[200:,max_pixel:,:] * arr_flip[None,...,None]
    # uv_texture[200:,max_pixel:,:] = torch.flip(uv_texture, (1,))[200:,max_pixel:,:] * arr[None,...,None] + uv_texture[200:,max_pixel:,:] * arr_flip[None,...,None]

    return uv_texture_c