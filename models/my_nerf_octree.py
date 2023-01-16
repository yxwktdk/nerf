import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import time
import skimage
import trimesh
import svox


def Valid(a, b, c, RS):
    return 0 <= a < RS and 0 <= b < RS and 0 <= c < RS


class CheatNeRF():
    def __init__(self, nerf):
        super(CheatNeRF, self).__init__()
        self.nerf = nerf

    def query(self, pts_xyz):
        return self.nerf(pts_xyz, torch.zeros_like(pts_xyz))


class MyNeRFoct():
    def __init__(self):
        super(MyNeRFoct, self).__init__()
        self.RS = 128
        RS = self.RS
        # 设置分辨率
        self.tree = svox.N3Tree(center=[0.0, 0.875, 0.0], radius=0.125).refine(6)
        #最初是128*128
        self.volume_sigma = torch.zeros((RS, RS, RS))
        self.volume_color = torch.zeros((RS, RS, RS, 3))
        # 存储体素数据结构是否还需要往下划分（01）
        self.FSTparent = np.zeros((RS, RS, RS))
        self.FSTp_num = 0

    def save(self, sigma, color):
        # 直接reshape之后存储
        RS = self.RS

        self.tree = torch.cat([color.reshape((RS * RS * RS, 3)),
                               sigma.reshape((RS * RS * RS, 1))],dim=1)
        self.volume_sigma = torch.zeros((RS, RS, RS))
        self.volume_color = torch.zeros((RS, RS, RS, 3))
        checkpoint = {
            "volume_sigma": self.volume_sigma,
            "volume_color": self.volume_color
        }
        torch.save(checkpoint, "temp.pth")


    def query(self, pts_xyz):
        N, _ = pts_xyz.shape
        sigma = torch.zeros(N, 1, device=pts_xyz.device)
        color = torch.zeros(N, 3, device=pts_xyz.device)
        # 利用归一化定位具体坐标
        RS = self.RS

        X_index = ((pts_xyz[:, 0] + 0.125) * 4 * RS).clamp(0, RS - 1).long()
        Y_index = ((pts_xyz[:, 1] - 0.75) * 4 * RS).clamp(0, RS - 1).long()
        Z_index = ((pts_xyz[:, 2] + 0.125) * 4 * RS).clamp(0, RS - 1).long()
        sigma[:, 0] = self.volume_sigma[X_index, Y_index, Z_index]
        color[:, :] = self.volume_color[X_index, Y_index, Z_index]

        return sigma, color

    def mcube(self, thres):
        vertices, faces, _, _ = skimage.measure.marching_cubes(self.volume_sigma.numpy(), thres)
        mesh = trimesh.Trimesh(vertices, faces)
        return mesh

    def FSTLayer(self, thres):
        # 根据体素数据结构（128*128*128）进行几何结构提取
        RS = self.RS
        vertices, faces, _, _ = skimage.measure.marching_cubes(self.volume_sigma.numpy(), thres)
        vertices = np.around(vertices).astype(np.int32)  # 顶点取整
        self.FSTp_num = 0

        for vertex in tqdm(vertices):
            # 遍历所有的顶点，将顶点所在的体素数据结构做好标记
            X_index = vertex[0] / (RS * 4) - 0.125
            Y_index = vertex[1] / (RS * 4) + 0.75
            Z_index = vertex[2] / (RS * 4) - 0.125


            if self.FSTparent[vertex[0]][vertex[1]][vertex[2]] == 0:
                self.FSTp_num = self.FSTp_num + 1
                self.FSTparent[vertex[0]][vertex[1]][vertex[2]] = self.FSTp_num

