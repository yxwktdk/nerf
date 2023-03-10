import os
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.fields import NeRF
from models.my_dataset import Dataset
from models.my_nerf import MyNeRF, CheatNeRF
from models.my_renderer import MyNerfRenderer
from models.my_nerf_octree import MyNeRFoct
import svox
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Runner:
    def __init__(self, conf_path, mode='render', case='CASE_NAME', is_continue=False):
        self.device = torch.device('cpu')
        # self.device = torch.device('cuda:0')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'], self.device)
        self.iter_step = 0
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')

        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        self.coarse_nerf = NeRF(**self.conf['model.coarse_nerf']).to(self.device)
        self.fine_nerf = NeRF(**self.conf['model.fine_nerf']).to(self.device)
        self.my_nerf = MyNeRF()
        self.my_nerf_oct = MyNeRFoct()
        if args.mode == 'oct':
            self.renderer = MyNerfRenderer(self.my_nerf_oct,
                                           **self.conf['model.nerf_renderer'])
        else:
            self.renderer = MyNerfRenderer(self.my_nerf,
                                           **self.conf['model.nerf_renderer'])

        self.load_checkpoint(r'D:\sophomore\algorithm\NeRF\NeRF\nerf_model.pth', absolute=True)

    def load_checkpoint(self, checkpoint_name, absolute=False):
        if absolute:
            checkpoint = torch.load(checkpoint_name, map_location=self.device)
        else:
            checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name),
                                    map_location=self.device)
        self.coarse_nerf.load_state_dict(checkpoint['coarse_nerf'])
        self.fine_nerf.load_state_dict(checkpoint['fine_nerf'])
        logging.info('End')

    def use_nerf(self):
        self.my_nerf = CheatNeRF(self.fine_nerf)
        self.renderer = MyNerfRenderer(self.my_nerf,
                                       **self.conf['model.nerf_renderer'])

    def save_oct(self):
        RS = 128
        checkpoint = torch.load("temp.pth")
        sigma = checkpoint["volume_sigma"]
        color = checkpoint["volume_color"]
        self.my_nerf_oct.save(sigma, color)
        FSTpars = self.my_nerf_oct.FSTLayer(args.mcube_threshold)
        '''checkpoint = {
            "tree":self.my_nerf_oct.tree
        }
        torch.save(checkpoint, "FSTtree.pth")'''
        SNDpars = self.my_nerf_oct.CheckVertex(FSTpars[1:], 0.25 / (RS * 2), args.mcube_threshold, self.fine_nerf)
        self.my_nerf_oct.CheckVertex(SNDpars[1:],0.25 / (RS * 4),args.mcube_threshold, self.fine_nerf,True)
        checkpoint = {
            "tree": self.my_nerf_oct.tree
        }
        torch.save(checkpoint, "tree.pth")

    def render_oct(self):
        images = []
        resolution_level = 4
        n_frames = 90
        # ?????????90?????????
        for idx in tqdm(range(n_frames)):
            rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
            H, W, _ = rays_o.shape
            rays_o = rays_o.reshape(-1, 3).split(1024)
            rays_d = rays_d.reshape(-1, 3).split(1024)

            out_rgb_fine = []

            # for rays_o_batch, rays_d_batch in tqdm(zip(rays_o, rays_d)):
            for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
                near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
                background_rgb = torch.ones([1, 3], device=self.device) if self.use_white_bkgd else None

                render_out = self.renderer.render_oct(rays_o_batch,
                                                  rays_d_batch,
                                                  near,
                                                  far,
                                                  background_rgb=background_rgb)

                def feasible(key):
                    return (key in render_out) and (render_out[key] is not None)

                if feasible('fine_color'):
                    out_rgb_fine.append(render_out['fine_color'].detach().cpu().numpy())

                del render_out

            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255)
            img_fine = cv.resize(cv.flip(img_fine, 0), (512, 512))
            images.append(img_fine)
            os.makedirs(os.path.join(self.base_exp_dir, 'render_oct'), exist_ok=True)
            cv.imwrite(os.path.join(self.base_exp_dir, 'render_oct', '{}.jpg'.format(idx)), img_fine)
        '''for idx in tqdm(range(n_frames)):
            ren = svox.VolumeRenderer(self.my_nerf_oct.tree)
            rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
            H, W, _ = rays_o.shape

            #rays_o = rays_o.reshape(-1, 3).split(1024)
            #rays_d = rays_d.reshape(-1, 3).split(1024)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            rays_v = torch.nn.functional.normalize(rays_d, dim=1)
            ray = svox.Rays(origins=rays_o, dirs=rays_d,
                            viewdirs=rays_v)
            render_out = ren(ray)
            out_rgb_fine = []
            out_rgb_fine.append(render_out.detach().cpu().numpy())
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255)
            img_fine = cv.resize(cv.flip(img_fine, 0), (512, 512))
            images.append(img_fine)

            os.makedirs(os.path.join(self.base_exp_dir, 'render_oct'), exist_ok=True)
            cv.imwrite(os.path.join(self.base_exp_dir, 'render_oct', '{}.jpg'.format(idx)), img_fine)

            for rays_o_batch, rays_d_batch in tqdm(zip(rays_o, rays_d)):
            #for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
                #near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
               # background_rgb = torch.ones([1, 3], device=self.device) if self.use_white_bkgd else None

                #render_out = self.renderer.render(rays_o_batch,rays_d_batch, near, far,background_rgb=background_rgb)
                ray = svox.Rays(origins=rays_o_batch, dirs = rays_d_batch, viewdirs = torch.nn.functional.normalize(rays_d_batch,dim = 1))
                render_out = ren(ray)
                out_rgb_fine.append(render_out.detach().cpu().numpy())

                del render_out

            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255)
            img_fine = cv.resize(cv.flip(img_fine, 0), (512, 512))
            images.append(img_fine)

            os.makedirs(os.path.join(self.base_exp_dir, 'render_oct'), exist_ok=True)
            cv.imwrite(os.path.join(self.base_exp_dir, 'render_oct', '{}.jpg'.format(idx)), img_fine)'''

    def save(self):
        RS = 128
        # ?????????64
        pts_xyz = torch.zeros((RS, RS, RS, 3))
        batch_size = 1024


        checkpoint = torch.load("temp.pth")
        sigma = checkpoint["volume_sigma"]
        color = checkpoint["volume_color"]



        self.my_nerf.save(pts_xyz, sigma, color)
        ''' for batch in tqdm(range(0, pts_xyz.shape[0], batch_size)):
                    batch_pts_xyz = pts_xyz[batch:batch+batch_size]
                    net_sigma, net_color = self.fine_nerf(batch_pts_xyz, torch.zeros_like(batch_pts_xyz))
                    sigma[batch:batch+batch_size] = net_sigma
                    color[batch:batch+batch_size] = net_color'''

    def render_video(self):
        images = []
        resolution_level = 4
        n_frames = 90
        # ?????????90?????????
        for idx in tqdm(range(n_frames)):
            rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
            H, W, _ = rays_o.shape
            rays_o = rays_o.reshape(-1, 3).split(1024)
            rays_d = rays_d.reshape(-1, 3).split(1024)

            out_rgb_fine = []

            # for rays_o_batch, rays_d_batch in tqdm(zip(rays_o, rays_d)):
            for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
                near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
                background_rgb = torch.ones([1, 3], device=self.device) if self.use_white_bkgd else None

                render_out = self.renderer.render(rays_o_batch,
                                                  rays_d_batch,
                                                  near,
                                                  far,
                                                  background_rgb=background_rgb)

                def feasible(key):
                    return (key in render_out) and (render_out[key] is not None)

                if feasible('fine_color'):
                    out_rgb_fine.append(render_out['fine_color'].detach().cpu().numpy())

                del render_out

            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255)
            img_fine = cv.resize(cv.flip(img_fine, 0), (512, 512))
            images.append(img_fine)
            os.makedirs(os.path.join(self.base_exp_dir, 'render'), exist_ok=True)
            cv.imwrite(os.path.join(self.base_exp_dir, 'render', '{}.jpg'.format(idx)), img_fine)

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(self.base_exp_dir, 'render', 'show.mp4'), fourcc, 30, (w, h))
        for image in tqdm(images):
            image = image.astype(np.uint8)
            writer.write(image)
        writer.release()


    def mcube(self):
        mesh = self.my_nerf.mcube(args.mcube_threshold)
        mesh.export('./exp/mcube/' + 'mcube{}.obj'.format(int(args.mcube_threshold)))
        '''        for i in range(30):
            thres = 3 * i - 45
            mesh = self.my_nerf.mcube(thres)
            sign =''
            if thres > 0:
                sign = '+'
            elif thres < 0:
                sign = '-'
                thres = 0 - thres
            mesh.export('./exp/mcube/' + sign + 'mcube{}.obj'.format(int(thres)))'''

    def FST(self):
        self.my_nerf.FSTLayer(args.mcube_threshold)


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='confs/nerf.conf')
    # parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='oct')
    #parser.add_argument('--mode', type=str, default='render')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--case', type=str, default='test')
    # parser.add_argument('--case', type=str, default='')
    parser.add_argument('--dataroot', type=str, default='')

    args = parser.parse_args()
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'render':
        runner.save()
        runner.render_video()
    elif args.mode == 'mcube':
        runner.save()
        runner.mcube()

    elif args.mode == 'test':
        runner.use_nerf()
        runner.render_video()

    elif args.mode == 'oct':
        runner.save_oct()
        runner.render_oct()
