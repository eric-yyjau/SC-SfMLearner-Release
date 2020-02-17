# KITIT_VO=/media/bjw/Disk/Dataset/kitti_odom/

KITIT_VO=/media/yoyee/Big_re/kitti/

POSE_NET=./pretrained/pose/cs+k_pose.tar

python test_pose.py $POSE_NET \
--img-height 256 --img-width 832 \
--dataset-dir $KITIT_VO \
--sequences 09