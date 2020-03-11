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


# 0305 experiments of sequence length: train on skipped frames
TRAIN_SET=/newfoundland/yyjau/kitti/scsfm_dump/kitti_vo_256/
python train.py $TRAIN_SET \
--dispnet DispResNet \
--num-scales 1 \
-b4 -s0.1 -c0.5 --epoch-size 1000 --sequence-length 5 \
--with-mask \
--with-ssim \
--skip_frame 2 \
--name posenet_256_wKeyframe \
--keyframe yes
# --pretrained-disp checkpoints/posenet_256_seqLen7/80/dispnet_model_best.pth.tar \
# --pretrained-pose  checkpoints/posenet_256_seqLen7/80/exp_pose_model_best.pth.tar \