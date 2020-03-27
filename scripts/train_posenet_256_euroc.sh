# 0324 Train on Euroc dataset
TRAIN_SET=/data/yyjau/euroc/euroc_dump/scsfm_fil_h240/
python train.py $TRAIN_SET \
--dispnet DispResNet \
--num-scales 1 \
-b4 -s0.1 -c0.5 --epoch-size 1000 --sequence-length 33 \
--with-mask \
--with-ssim \
--skip_frame 8 \
--pose_train \
--name posenet_256_euroc_v0.2_sk8_5fr \
--pretrained-disp checkpoints/posenet_256/156/dispnet_model_best.pth.tar \
--pretrained-pose  checkpoints/posenet_256/156/exp_pose_model_best.pth.tar \
--notes "use pretrained model skip 8, freeze dispnet"

# --keyframe ./datasets/kitti_keyframe/orbslam2_key/
