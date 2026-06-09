"""
Util functions for network construction
"""
import os
import importlib
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def coords_grid(b, h, w, device):
    coords = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(b, 1, 1, 1)

def backward_warp(img, flow, pad='zeros'):
    b, c, h, w = img.shape
    grid = coords_grid(b, h, w, device=img.device)
    grid = grid + flow
    xgrid, ygrid = grid.split([1,1], dim=1)
    xgrid = 2*xgrid/(w-1) - 1
    ygrid = 2*ygrid/(h-1) - 1
    grid = torch.cat([xgrid, ygrid], dim=1)
    warped_img = F.grid_sample(input=img, grid=grid.permute(0,2,3,1), mode='bilinear',  padding_mode='zeros') 
    return warped_img

def resize_flow(flow, h, w):
    b, c, c_h, c_w = flow.shape
    flow = F.interpolate(input=flow, size=(h, w), mode='bilinear', align_corners=False)
    flow[:, 0] *= float(w) / float(c_w) # rescale flow magnitude after rescaling spatial size
    flow[:, 1] *= float(h) / float(c_h)
    return flow

def resize_img_to_factor_of_k(img, k=64, mode='bilinear'):
    b, c, h, w = img.shape
    new_h = int(np.ceil(h / float(k)) * k)
    new_w = int(np.ceil(w / float(k)) * k)
    img = F.interpolate(input=img, size=(new_h, new_w), mode=mode, align_corners=False)
    return img

def pad_img_to_factor_of_k(img, k=64, mode='replicate'):
    if img.ndimension() == 4:
        b, c, h, w = img.shape
        pad_h, pad_w = k - h % k, k - w % k
        img = F.pad(img, (0, pad_w, 0, pad_h), mode='replicate')
    elif img.ndimension() == 5:
        h, w = img.shape[-2], img.shape[-1]
        pad_h, pad_w = k - h % k, k - w % k
        img = F.pad(img, (0, pad_w, 0, pad_h, 0, 0), mode='replicate')
    return img

