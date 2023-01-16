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
        self.tree = svox.N3Tree(center=[0.0, 0.875, 0.0], radius=0.125)
        self.tree.refine(6)
        #最初是128*128
        self.volume_sigma = torch.zeros((RS, RS, RS))
        self.volume_color = torch.zeros((RS, RS, RS, 3))
        # 存储体素数据结构是否还需要往下划分（01）
        self.FSTpars = torch.tensor([[0,0,0]])
        self.SNDpars = torch.tensor([[0, 0, 0]])
        self.FSTp_num = 0

    def save(self, sigma, color):
        # 直接reshape之后存储
        RS = self.RS
        self.tree[:] = torch.cat([color.reshape((RS * RS * RS, 3)),
                               sigma.reshape((RS * RS * RS, 1))],dim=1)
        self.volume_sigma = torch.zeros((RS, RS, RS))
        self.volume_color = torch.zeros((RS, RS, RS, 3))
        self.volume_sigma = sigma

        self.volume_color = color
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

    def CheckVertex(self, vertices_o, radius, thres, model, last_refine = False):
        """
        对于一个给定点，查看该位置八个点的对应的sigma和color，
        先把color和sigma输入到八个叶子当中，然后利用sigma进行几何重建
        :param vertices_o: 作为查看的坐标原点
        :param radius: 查看的每一步的跨度（半径）
        :param thres: 几何重建的阈值
        :param model: 传入模型
        :param last_refine: 是不是最后一层，如果是的话不再寻找顶点
        :return:
        """
        depth = 0
        r = radius
        for i in range(10):
            if r != 0.25:
                r*=2
                depth+=1
        for vertex_o in tqdm(vertices_o[1:]):
            pts_xyz = torch.tensor([[[[vertex_o[0], vertex_o[1], vertex_o[2]], [vertex_o[0], vertex_o[1], vertex_o[2] + radius]],
                                     [[vertex_o[0], vertex_o[1] + radius, vertex_o[2]],
                                      [vertex_o[0], vertex_o[1] + radius, vertex_o[2] + radius]]],
                                    [[[vertex_o[0] + radius, vertex_o[1], vertex_o[2]],
                                      [vertex_o[0] + radius, vertex_o[1], vertex_o[2] + radius]],
                                     [[vertex_o[0] + radius, vertex_o[1] + radius, vertex_o[2]],
                                      [vertex_o[0] + radius, vertex_o[1] + radius, vertex_o[2] + radius]]]])
            pts_xyz = pts_xyz.reshape(8, 3)

            sigma, color = model(pts_xyz, torch.zeros_like(pts_xyz))
            self.tree[pts_xyz] = torch.cat([color, sigma], dim=1)#更新叶子结点数据
            if thres > max(sigma) or thres < min(sigma):
                continue
            if not last_refine:
                vertices, _, _, _ = skimage.measure.marching_cubes(sigma.reshape((2, 2, 2)).numpy(), thres)
                vertices = np.around(vertices)

                vertices[:, 0] = vertices[:, 0] * radius + vertex_o[0]
                vertices[:, 1] = vertices[:, 1] * radius + vertex_o[1]
                vertices[:, 2] = vertices[:, 2] * radius + vertex_o[2]
                for vertex in vertices:
                    if self.tree[vertex[0], vertex[1], vertex[2]].depths == depth - 1:
                        # 表示还没有被refine
                        self.FSTp_num += 1
                        self.tree[vertex[0], vertex[1], vertex[2]].refine()
                        vertex = np.ascontiguousarray(vertex)
                        self.SNDpars = torch.cat((self.SNDpars, torch.tensor(vertex).reshape(1, 3)), dim=0)
        return self.SNDpars

    def FSTLayer(self, thres):

        """
        根据体素数据结构（128*128*128）进行几何结构提取
        提取出来的顶点保存在vertices中，然后将tree对应的进行refine
        将所有进一步拆分的点存入FSTpars中，作为走模型的点
        :param thres: 几何结构提取的阈值
        :return 返回第一层的顶点
        """
        RS = self.RS
        vertices, _, _, _ = skimage.measure.marching_cubes(self.volume_sigma.numpy(), thres)
        self.FSTp_num = 0
        vertices = np.around(vertices)

        vertices[:, 0] = vertices[:, 0] / (RS * 4) - 0.125
        vertices[:, 1] = vertices[:, 1] / (RS * 4) + 0.75
        vertices[:, 2] = vertices[:, 2] / (RS * 4) - 0.125

        for vertex in tqdm(vertices):
            # 遍历所有的顶点，将顶点所在的体素数据结构做好标记
            #为了减少标记点，或许可以把临近点只标注一个？
            if self.FSTp_num == 100:
                return self.FSTpars#调试过程中
            if self.tree[vertex[0], vertex[1], vertex[2]].depths == 6:
                #表示还没有被refine
                self.FSTp_num+=1
                self.tree[vertex[0], vertex[1], vertex[2]].refine()
                vertex = np.ascontiguousarray(vertex)
                self.FSTpars = torch.cat((self.FSTpars, torch.tensor(vertex).reshape(1,3)),dim=0)

        return self.FSTpars

