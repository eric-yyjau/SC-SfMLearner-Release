import torch
from imageio import imread, imsave
from skimage.transform import resize as imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

from inverse_warp import *
from scipy.ndimage.interpolation import zoom

import models
import cv2

parser = argparse.ArgumentParser(description='Script for visualizing depth map and masks',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-posenet", required=True,
                    type=str, help="pretrained PoseNet path")
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=832, type=int, help="Image width")
parser.add_argument("-d", "--dataset", default='kitti', type=str, help="dataset type")
parser.add_argument("--dataset-dir", type=str, help="Dataset directory")
parser.add_argument("--output-dir", type=str,
                    help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'],
                    nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--sequence", default='09',
                    type=str, help="sequence to test")
parser.add_argument("--save_video", action="store_true", help="save as video")
parser.add_argument("--skip_frame", default=1, type=int, help="The time differences between frames")
parser.add_argument("--keyframe", default="", type=str, help="File with keyframe stamps")
parser.add_argument("--all_frame", action="store_true", help="export all frames based on keyframes")
parser.add_argument("--lstm", action='store_true', default=False, help="use lstm network")

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

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

def load_tensor_image(filename, args):
    img = imread(filename).astype(np.float32)
    if img.ndim == 2:
        img = np.tile(img[..., np.newaxis], (1, 1, 3)) # expand to rgb
    h, w, _ = img.shape
    if (h != args.img_height or w != args.img_width):
        img = imresize(img, (args.img_height, args.img_width)
                       ).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    tensor_img = (
        (torch.from_numpy(img).unsqueeze(0)/255 - 0.5)/0.5).to(device)
    return tensor_img


@torch.no_grad()
def main():
    args = parser.parse_args()
    print(f"args: {args}")

    # model
    weights_pose = torch.load(args.pretrained_posenet)
    lstm = args.lstm
    if lstm:
        from models.PoseLstmNet import PoseLstmNet
        pose_net = PoseLstmNet().to(device)
    else:
        pose_net = models.PoseNet().to(device)
    pose_net.load_state_dict(weights_pose['state_dict'], strict=False)
    pose_net.eval()

    # output dir
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    # dataset
    if args.dataset == 'kitti':
        image_dir = Path(args.dataset_dir + args.sequence + "/image_2/")
    elif args.dataset == 'euroc':
        # subfolders = '/datasets/euroc/V1_01_easy/mav0/cam0/data/'
        subfolders = '/mav0/'
        image_dir = Path(args.dataset_dir + args.sequence + subfolders)

    if args.dataset == 'kitti':
        test_files = sum([image_dir.files('*.{}'.format(ext))
                      for ext in args.img_exts], [])
    elif args.dataset == 'euroc':
        test_files = read_images_files_from_folder(
                image_dir, folder="cam0"
        )
        pass
    test_files.sort()
    print('{} files to test'.format(len(test_files)))

    ## init pose
    global_pose = np.identity(4)
    poses = [global_pose[0:3, :].reshape(1, 12)]

    ## load the first image
    n = len(test_files)
    tensor_img1 = load_tensor_image(test_files[0], args)

    ## for saving video
    if args.save_video:
        width = args.width
        height = args.height
        FPS = 24
        fourcc = VideoWriter_fourcc(*'MP4V')
        video = VideoWriter(f'{args.output_dir}/args.sequence/demo.mp4', fourcc, float(FPS), (width, height))        

    skip_frame = args.skip_frame
    time_stamps = [0]
    # check if there's keyframe file
    if args.keyframe != "":
        from utils import load_keyframe
        kf_arr = load_keyframe(args.keyframe)
        print(f"keyframe length: {len(kf_arr)}, percent: {len(kf_arr)/n}")
    else:
        kf_arr = range(0, n - skip_frame, skip_frame)
    
    # use all frames for estimation
    if args.all_frame:
        # assert args.skip_frame == 1
        loop_arr = range(0, n - 1)
        # make sure keyframe covers all frames
        kf_arr = np.concatenate((np.arange(kf_arr[0]), kf_arr, np.arange(kf_arr[-1]+1, n)), axis=0)
        key_pose = global_pose
    else:
        loop_arr = kf_arr
    # print(f"loop_arr: {loop_arr}")

    # init lstm hidden layer
    if lstm:
        pose_net.init_lstm_states(tensor_img1)

    idx_kf = 0
    tensor_img2 = None
    reset_interval = -1
    print(f"+++++ reset interval: {reset_interval} +++++")
    idx_img2 = -1
    for iter in tqdm(loop_arr):
        if args.all_frame:
            # point to the correct idx. if idx moves, load image again
            if kf_arr[idx_kf] <= iter and iter < kf_arr[idx_kf+1]:
                # keep this frame
                # print(f"keep frame: {kf_arr[idx_kf]}, iter: {iter}")
                pass
            else:
                idx_kf += 1
                # assert kf_arr[idx_kf] < iter
                # print(f"update frame: {kf_arr[idx_kf]}, iter: {iter}")
                # load image 1
                idx_img1 = kf_arr[idx_kf]
                if idx_img1 == idx_img2:
                    tensor_img1 = tensor_img2
                else:
                    tensor_img1 = load_tensor_image(test_files[idx_img1], args)
                # update key_pose
                key_pose = global_pose
        if reset_interval > 0 and iter%reset_interval == 0:
            pose_net.init_lstm_states(tensor_img1)
            pose = pose_net(tensor_img1)
        # same process
        ## load image2
        idx_img2 = iter + 1
        if args.all_frame:
            # step = 1 for all frames
            tensor_img2 = load_tensor_image(test_files[iter+1], args)
        else:
            tensor_img2 = load_tensor_image(test_files[iter+skip_frame], args)
            idx_img2 = iter + skip_frame

        if lstm:
            pose = pose_net(tensor_img2)
        else:
            pose = pose_net(tensor_img1, tensor_img2)
        pose_mat = pose_vec2mat(pose).squeeze(0).cpu().numpy()
        pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])
        
        if args.all_frame:
            global_pose = key_pose @ np.linalg.inv(pose_mat)
            poses.append(global_pose[0:3, :].reshape(1, 12))
            time_stamps.append(iter+1)
        else:
            global_pose = global_pose @ np.linalg.inv(pose_mat)
            poses.append(global_pose[0:3, :].reshape(1, 12))
            time_stamps.append(iter+skip_frame)

            # update
            tensor_img1 = tensor_img2


    poses = np.concatenate(poses, axis=0)
    time_stamps = np.array(time_stamps).reshape(-1, 1)
    poses_wTime = np.concatenate((time_stamps, poses), axis=1)
    # save to files
    filename = Path(args.output_dir + args.sequence + "_noTime.txt")
    np.savetxt(filename, poses, delimiter=' ', fmt='%1.8e')
    filename = Path(args.output_dir + args.sequence + "_t.txt")
    np.savetxt(filename, poses_wTime, delimiter=' ', fmt='%1.8e')
    filename = Path(args.output_dir + args.sequence + ".txt") # save the time as default
    np.savetxt(filename, poses_wTime, delimiter=' ', fmt='%1.8e')


if __name__ == '__main__':
    main()
