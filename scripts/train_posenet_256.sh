# TRAIN_SET=/newfoundland/yyjau/kitti/scsfm_dump/kitti_vo_256/
# python train.py $TRAIN_SET \
# --dispnet DispResNet \
# --num-scales 1 \
# -b4 -s0.1 -c0.5 --epoch-size 1000 --sequence-length 3 \
# --with-mask \
# --with-ssim \
# --name posenet_256


# experiments of sequence length
# TRAIN_SET=/newfoundland/yyjau/kitti/scsfm_dump/kitti_vo_256/
# python train.py $TRAIN_SET \
# --dispnet DispResNet \
# --num-scales 1 \
# -b4 -s0.1 -c0.5 --epoch-size 1000 --sequence-length 5 \
# --with-mask \
# --with-ssim \
# --name posenet_256_seqLen7


# 0305 experiments of sequence length: train on keyframe files
TRAIN_SET=/newfoundland/yyjau/kitti/scsfm_dump/kitti_vo_256/
python train.py $TRAIN_SET \
--dispnet DispResNet \
--num-scales 1 \
-b4 -s0.1 -c0.5 --epoch-size 1000 --sequence-length 5 \
--with-mask \
--with-ssim \
--skip_frame 1 \
--name posenet_256_kf_orbslam2 \
--keyframe ./datasets/kitti_keyframe/orbslam2_key/
# --keyframe ./datasets/kitti_keyframe/rot_thd0.5/
# --pretrained-disp checkpoints/posenet_256_seqLen7/80/dispnet_model_best.pth.tar \
# --pretrained-pose  checkpoints/posenet_256_seqLen7/80/exp_pose_model_best.pth.tar \