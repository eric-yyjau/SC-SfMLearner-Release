# 0305 experiments of sequence length: train on skipped frames
TRAIN_SET=/newfoundland/yyjau/kitti/scsfm_dump/kitti_vo_256/
python train.py $TRAIN_SET \
--dispnet DispResNet \
--num-scales 1 \
-b4 -s0.1 -c0.5 --epoch-size 1000 --sequence-length 11 \
--with-mask \
--with-ssim \
--name poselstmnet_256_v0.2_2fr \
--batch-size 2 \
--lstm \
--dataParallel \
# --debug \
# --keyframe ./datasets/kitti_keyframe/orbslam2_key/
# --pretrained-disp checkpoints/posenet_256_seqLen7/80/dispnet_model_best.pth.tar \
# --pretrained-pose  checkpoints/posenet_256_seqLen7/80/exp_pose_model_best.pth.tar \