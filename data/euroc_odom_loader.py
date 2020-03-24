from __future__ import division
import numpy as np
# from path import Path
from pathlib import Path
import scipy.misc
from collections import Counter
import os
from kitti_odom_loader import KittiOdomLoader
from glob import glob

class EurocOdomLoader(KittiOdomLoader):
    def __init__(self,
                 dataset_dir,
                 img_height=256,
                 img_width=832):

        self.dataset_dir = Path(dataset_dir)
        self.img_height = img_height
        self.img_width = img_width
        # self.cam_ids = ['2', '3']
        self.cam_ids = ['0', '1']
        # self.train_sets = ["MH_01_easy"]
        self.train_sets = [
                    "MH_01_easy",
                    "MH_02_easy",
                    "MH_04_difficult",
                    "V1_01_easy",
                    "V1_02_medium",
                    "V1_03_difficult",
                    ]
        self.test_sets = [
                    "MH_02_easy",
                    "MH_05_difficult",
                    "V2_01_easy",
                    "V2_02_medium",
                    "V2_03_difficult",
                    ]

        self.collect_train_folders()

    def collect_train_folders(self, subdir='.'):
        self.scenes = []
        sequence_list = [x for x in (self.dataset_dir/subdir).iterdir() if x.is_dir()]
        # sequence_list = (self.dataset_dir/subdir).dirs()
        for sequence in sequence_list:
            if sequence.name in self.train_sets:
                self.scenes.append(sequence)

    def get_cam_rel_path(self, scene_data):
        return Path(scene_data['dir'])/'mav0'/f"cam{scene_data['cid']}"

    def collect_scenes(self, drive):
        train_scenes = []
        for c in self.cam_ids:
            scene_data = {'cid': c, 'dir': drive, 'frame_id': [], 'rel_path': drive.name + '_' + c}
            
            img_dir = Path(scene_data['dir']/'mav0'/f"cam{scene_data['cid']}/data")
            # scene_data['frame_id'] = [x.split('.')[0] for x in os.listdir(img_dir)]
            scene_data['frame_id'] = glob(str(img_dir) + "/*")  # path to the images
            print(f"img_dir: {img_dir}")
            print(f"scene_data['frame_id']: {len(scene_data['frame_id'])}")

            sample, zoom_x, zoom_y = self.load_image(scene_data, 0)
            if sample is None:
                print(f"sample is none")
                return []

            scene_data['intrinsics'] = self.read_calib_file(c, scene_data, zoom_x, zoom_y)
            train_scenes.append(scene_data)

        return train_scenes

    def get_scene_imgs(self, scene_data):
        for (i,frame_id) in enumerate(scene_data['frame_id']):
            yield {"img":self.load_image(scene_data, i)[0], "id":Path(frame_id).stem}

    def load_image(self, scene_data, tgt_idx):
        # img_file = scene_data['dir']/'cam{}'.format(scene_data['cid'])/scene_data['frame_id'][tgt_idx]+'.png'
        img_file = scene_data['frame_id'][0]
        if not Path(img_file).is_file():
            return None
        img = scipy.misc.imread(img_file)
        zoom_y = self.img_height/img.shape[0]
        zoom_x = self.img_width/img.shape[1]
        img = scipy.misc.imresize(img, (self.img_height, self.img_width))
        return img, zoom_x, zoom_y

    def read_calib_file(self, cid, scene_data, zoom_x, zoom_y):
        # with open(filepath, 'r') as f:
        #     C = f.readlines()
        # def parseLine(L, shape):
        #     data = L.split()
        #     data = np.array(data[1:]).reshape(shape).astype(np.float32)
        #     return data
        # proj_c2p = parseLine(C[int(cid)], shape=(3,4))
        # calib_file = scene_data['dir']/'cam{}'.format(scene_data['cid'])/'sensor.yaml'
        calib_file = self.get_cam_rel_path(scene_data)/'sensor.yaml'
        print(f"calib_data: {calib_file}")
        calib_data = loadConfig(calib_file)
        height, width, calib, D = self.load_intrinsics(calib_data)
        # calib = proj_c2p[0:3, 0:3]
        calib[0,:] *=  zoom_x
        calib[1,:] *=  zoom_y

        return calib

    @staticmethod
    def load_intrinsics(calib_data): # for euroc
        width, height = calib_data["resolution"]
        # cam_info.distortion_model = 'plumb_bob'
        D = np.array(calib_data["distortion_coefficients"])
        # cam_info.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        fu, fv, cu, cv = calib_data["intrinsics"]
        K = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]])

        return height, width, K, D
 

def loadConfig(filename):
    import yaml
    with open(filename, "r") as f:
        config = yaml.load(f)
    return config