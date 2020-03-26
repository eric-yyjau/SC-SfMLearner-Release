from __future__ import division
import shutil
import numpy as np
import torch
from path import Path
import datetime
from collections import OrderedDict
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


# from utils import load_keyframe
def load_keyframe(file):
    stamps = np.genfromtxt(file, dtype=float)
    if np.ndim(stamps) == 2:
        stamps = stamps[:,:1]
    stamps = stamps.reshape(-1,) - 1 # to offset the counting system starting from 1
    loop_arr = stamps.astype(int)
    return loop_arr

def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0, 1, low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0, max_value, resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:, i])
                         for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )

    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


COLORMAPS = {'rainbow': opencv_rainbow(),
             'magma': high_res_colormap(cm.get_cmap('magma')),
             'bone': cm.get_cmap('bone', 10000)}


def tensor2array(tensor, max_value=None, colormap='rainbow'):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        norm_array = tensor.squeeze().numpy()/max_value
        array = COLORMAPS[colormap](norm_array).astype(np.float32)
        array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        array = 0.5 + tensor.numpy()*0.5
    return array


def save_checkpoint(save_path, dispnet_state, exp_pose_state, is_best, filename='checkpoint.pth.tar'):
    file_prefixes = ['dispnet', 'exp_pose']
    states = [dispnet_state, exp_pose_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path/'{}_{}'.format(prefix, filename))

    if is_best:
        for prefix in file_prefixes:
            shutil.copyfile(save_path/'{}_{}'.format(prefix, filename),
                            save_path/'{}_model_best.pth.tar'.format(prefix))


####### functions in test_*.py #######

# from utils import read_images_files_from_folder
def read_images_files_from_folder(drive_path, folder="rgb"):
    # print(f"cid_num: {scene_data['cid_num']}")
    # img_dir = os.path.join(drive_path, "cam%d" % scene_data["cid_num"])
    # img_files = sorted(glob(img_dir + "/data/*.png"))
    print(f"drive_path: {drive_path}")
    ## given that we have matched time stamps
    arr = np.genfromtxt(
        f"{drive_path}/{folder}/data_f.txt", dtype="str"
    )  # [N, 2(time, path)]
    img_files = np.char.add(str(drive_path) + f"/{folder}/data/", arr[:, 1])
    img_files = [Path(f) for f in img_files]
    img_files = sorted(img_files)

    print(f"img_files: {img_files[0]}")
    return img_files

# from utils import load_tensor_image
from imageio import imread
from skimage.transform import resize as imresize

def load_tensor_image(filename, args, device="cpu"):
    img = imread(filename).astype(np.float32)
    if img.ndim == 2:
        img = np.tile(img[..., np.newaxis], (1, 1, 3))  # expand to rgb
    h, w, _ = img.shape
    if h != args.img_height or w != args.img_width:
        img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).unsqueeze(0) / 255 - 0.5) / 0.5).to(device)
    return tensor_img