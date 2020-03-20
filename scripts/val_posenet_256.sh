
# 0305 experiments of sequence length: train on skipped frames
TRAIN_SET=/media/yoyee/Big_re/kitti/scsfm_dump/kitti_vo_256
python val.py $TRAIN_SET \
--dispnet DispResNet \
--num-scales 1 \
-b1 -s0.1 -c0.5 --epoch-size 1 --sequence-length 3 \
--with-mask \
--with-ssim \
--skip_frame 1 \
--name val_posenet_test \
# --keyframe ./datasets/kitti_keyframe/orbslam2_key/
# --pretrained-disp checkpoints/posenet_256_seqLen7/80/dispnet_model_best.pth.tar \
# --pretrained-pose  checkpoints/posenet_256_seqLen7/80/exp_pose_model_best.pth.tar \