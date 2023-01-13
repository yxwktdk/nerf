import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import time

def Valid(a, b, c, RS):
    return 0 <= a < RS and 0 <= b < RS and 0 <= c < RS
class CheatNeRF():
    def __init__(self, nerf):
        super(CheatNeRF, self).__init__()
        self.nerf = nerf
    
    def query(self, pts_xyz):
        return self.nerf(pts_xyz, torch.zeros_like(pts_xyz))

class MyNeRF():
    def __init__(self):
        super(MyNeRF, self).__init__()
        self.RS = 128
        RS = self.RS
        #设置分辨率
        self.volume_sigma = torch.zeros((RS, RS, RS))
        self.volume_color = torch.zeros((RS, RS, RS, 3))

    def save(self, pts_xyz, sigma, color):
        #直接reshape之后存储
        RS = self.RS
        self.volume_sigma = sigma.reshape((RS, RS, RS))
        self.volume_color = color.reshape((RS, RS, RS, 3))
        checkpoint = {
            "volume_sigma": self.volume_sigma,
            "volume_color": self.volume_color
        }
        torch.save(checkpoint, "temp.pth")
        pass
    
    def query(self, pts_xyz):
        N, _ = pts_xyz.shape
        sigma = torch.zeros(N, 1, device=pts_xyz.device)
        color = torch.zeros(N, 3, device=pts_xyz.device)
        #利用归一化定位具体坐标
        RS = self.RS

        X_index = ((pts_xyz[:, 0] + 0.125) * 4 * RS).clamp(0, RS - 1).long()
        Y_index = ((pts_xyz[:, 1] - 0.75) * 4 * RS).clamp(0, RS - 1).long()
        Z_index = ((pts_xyz[:, 2] + 0.125) * 4 * RS).clamp(0, RS - 1).long()
        sigma[:, 0] = self.volume_sigma[X_index, Y_index, Z_index]
        color[:, :] = self.volume_color[X_index, Y_index, Z_index]

        return sigma, color


