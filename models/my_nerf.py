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
        print(pts_xyz.shape)
        print(sigma.shape)
        print(color.shape)
        #直接reshape之后存储
        RS = self.RS
        self.volume_sigma = sigma.reshape((RS, RS, RS))
        self.volume_color = color.reshape((RS, RS, RS, 3))

        pass
    
    def query(self, pts_xyz):
        N, _ = pts_xyz.shape
        sigma = torch.zeros(N, 1, device=pts_xyz.device)
        color = torch.zeros(N, 3, device=pts_xyz.device)
        #利用归一化定位具体坐标
        RS = self.RS
        x_time = 0.125 * 2 / RS
        y_time = (1.0 - 0.75) / RS
        z_time = 0.125 * 2 / RS

        turn_xyz = pts_xyz

        turn_xyz[:, 0] = (turn_xyz[: , 0] + 0.125) / x_time
        turn_xyz[:, 1] = (turn_xyz[:, 1] - 0.75) / y_time
        turn_xyz[:, 2] = (turn_xyz[:, 2] + 0.125) / z_time
        #turn_xyz are points after scaling and shifting
        turn_xyz = torch.round(turn_xyz)
        turn_xyz = turn_xyz.long()
        origin = torch.tensor((0,0,0))
        for i in range(N):
            turn_xyz[i] = turn_xyz[i]if Valid(turn_xyz[i,0], turn_xyz[i,1], turn_xyz[i,2], RS)else origin
            #把所有超过范围[0-64,0-64,0-64]的点变成[0,0,0]
        sigma = self.volume_sigma[turn_xyz[:, 0],turn_xyz[:, 1],turn_xyz[:, 2]]
        color = self.volume_color[turn_xyz[:, 0],turn_xyz[:, 1],turn_xyz[:, 2]]
        '''        for i in range(N):
            if Valid(turn_xyz[i,0], turn_xyz[i,1], turn_xyz[i,2], RS):
                sigma[i] = self.volume_sigma[turn_xyz[i,0]][turn_xyz[i,1]][turn_xyz[i,2]]
                color[i] = self.volume_color[turn_xyz[i,0]][turn_xyz[i,1]][turn_xyz[i,2]]
                count = count + 1
            else:
                sigma[i] = 0
                color[i] = torch.tensor([0, 0, 0])'''


        return sigma, color


