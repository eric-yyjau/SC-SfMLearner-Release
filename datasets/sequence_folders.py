import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random


def load_as_float(path):
    return imread(path).astype(np.float32)


class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, target_transform=None, skip_frame=1, keyframe=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.crawl_folders(sequence_length, skip_frame, keyframe)

    def crawl_folders(self, sequence_length, skip_frame=1, keyframe=None):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length, demi_length + 1, skip_frame))
        if keyframe is not None:
            from utils import load_keyframe
            kf_arr = load_keyframe(keyframe)
            pass
        # print(f"shifts: {shifts}")
        shifts.pop(demi_length)
        for scene in self.scenes:
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            imgs = sorted(scene.files('*.jpg'))
            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length, len(imgs)-demi_length):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                # keep an eye in the keyframe list
                idx_kf = 1
                kf_end = len(kf_arr)
                # regular add frames
                if i > kf_arr[0] and i < kf_arr[-1]:
                    while (idx_kf<kf_end):
                        if i < kf_arr[idx_kf]:
                            break
                        else:
                            idx_kf += 1
                    tmp_shifts = list(range(kf_arr[idx_kf-1]-i, kf_arr[idx_kf]-i))
                else:
                    tmp_shifts = shifts
                print(f"tmp_shifts: {tmp_shifts}")

                for j in tmp_shifts:
                    sample['ref_imgs'].append(imgs[i+j])

                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        return len(self.samples)
