# 0324 Train on Euroc dataset
TRAIN_SET=/data/yyjau/euroc/euroc_dump/scsfm_h240/
python train.py $TRAIN_SET \
--dispnet DispResNet \
--num-scales 1 \
-b4 -s0.1 -c0.5 --epoch-size 1000 --sequence-length 5 \
--with-mask \
--with-ssim \
--skip_frame 1 \
--name posenet_256_euroc_v0.1 \
# --keyframe ./datasets/kitti_keyframe/orbslam2_key/
# --pretrained-disp checkpoints/posenet_256_seqLen7/80/dispnet_model_best.pth.tar \
# --pretrained-pose  checkpoints/posenet_256_seqLen7/80/exp_pose_model_best.pth.tar \