import torch
from skimage.transform import resize as imresize
from imageio import imread
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
import models

parser = argparse.ArgumentParser(description='Script for DispNet testing with corresponding groundTruth',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-dispnet", required=True,
                    type=str, help="pretrained DispNet path")
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=832, type=int, help="Image width")
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=80)
parser.add_argument("-d", "--dataset", default='kitti', type=str, help="dataset type")
parser.add_argument("--dataset-dir", default='.',
                    type=str, help="Dataset directory")
# parser.add_argument("--dataset-list", default=None,
#                     type=str, help="Dataset list file")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'],
                    nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--output-dir", default=None, required=True, type=str,
                    help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument('--dispnet', dest='dispnet', required=True, type=str, choices=['DispNet', 'DispResNet'],
                    help='depth network architecture.')
parser.add_argument("--sequence", default='09',
                    type=str, help="sequence to test")
parser.add_argument("--save_video", action="store_true", help="save as video")

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_img(filename, args):
    img = imread(filename).astype(np.float32)
    h, w, _ = img.shape
    if (h != args.img_height or w != args.img_width):
        img = imresize(img, (args.img_height, args.img_width)
                       ).astype(np.float32)    
    return img

def load_tensor_image(filename, args):
    img = load_img(filename, args)
    img = np.transpose(img, (2, 0, 1))
    tensor_img = (
        (torch.from_numpy(img).unsqueeze(0)/255 - 0.5)/0.5).to(device)
    return tensor_img

def read_depth_npy(filename):
    depth_arr = np.load(filename)
    return depth_arr

@torch.no_grad()
def main():
    args = parser.parse_args()

    disp_net = getattr(models, args.dispnet)().to(device)
    weights = torch.load(args.pretrained_dispnet)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()
    # dataset_dir = Path(args.dataset_dir)
    # with open(args.dataset_list, 'r') as f:
    #     test_files = list(f.read().splitlines())
    # print('{} files to test'.format(len(test_files)))

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

    ## load the first image
    n = len(test_files)
    tensor_img1 = load_tensor_image(test_files[0], args)

    ## for saving video
    if args.save_video:
        from cv2 import VideoWriter, VideoWriter_fourcc
        width = args.img_width
        height = args.img_height*2
        FPS = 24
        fourcc = VideoWriter_fourcc(*'MP4V')
        output_video = f'{output_dir}/demo.mp4'
        print(f"save video: {output_video}")
        video = VideoWriter(output_video, fourcc, float(FPS), (width, height))        

    # for j in tqdm(range(len(test_files))):
    for iter in tqdm(range(n - 1)):
        tgt_img = load_tensor_image(test_files[iter+1], args)
        # tgt_img = load_tensor_image(dataset_dir + test_files[j], args)
        pred_disp = disp_net(tgt_img).cpu().numpy()[0, 0]

        j = iter
        if j == 0:
            predictions = np.zeros((len(test_files), *pred_disp.shape))
        depth = 1/pred_disp
        predictions[j] = depth

        if args.save_video:
            gray2rbg = lambda a: np.tile(a[:,:,None], (1,1,3))
            img_np = load_img(test_files[iter+1], args)
            def depth_post_processing(depth):
                depth = depth/(depth.max()+1e-6)
                depth = np.clip(depth, 0, 1)
                depth = (depth*255).astype(np.int32)
                return depth
            depth = depth_post_processing(depth)
            depth_rgb = gray2rbg(depth)
            # print(f"img_np: {img_np.max()}, depth_rgb: {depth_rgb.max()}")

            frame = np.concatenate((img_np, depth_rgb), axis=0).astype(np.uint8)
            # print(f"frame: {frame.shape}")
            video.write(frame)
        # if iter > 10:
        #     break
    
    np.save(output_dir/'predictions.npy', predictions)
    if args.save_video:
        video.release()


if __name__ == '__main__':
    main()
