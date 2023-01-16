import torch
import svox

tree=svox.N3Tree(center=[0.0, 0.875, 0.0], radius=0.125)
tree.refine(6)

#t是size(8,4)的一个数据结构
print(tree[0,0,0])
a =0
for i in range(8):
    tree[i] = 0
    tree[i] = a=a+1
b=torch.tensor([[0.1026, -0.3812, -0.9188],[ 0.1026, -0.3812, -0.9188]])
a=torch.tensor([[0.0210, 1.0595, 0.8249],[0.0210, 1.0595, 0.8249]])

ren = svox.VolumeRenderer(tree)
ray = svox.Rays(a,b,b)
print(ren(ray))

