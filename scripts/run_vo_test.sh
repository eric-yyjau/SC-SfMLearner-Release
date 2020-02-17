seq=10

DATASET_DIR=/media/yoyee/Big_re/kitti/sequences/
OUTPUT_DIR=results/vo/cs+k_pose/$seq

POSE_NET=./pretrained/pose/cs+k_pose.tar

# save the visual odometry results to "results_dir/09.txt"

#python test_vo.py \
#--img-height 256 --img-width 832 \
#--sequence $seq \
#--pretrained-posenet $POSE_NET --dataset-dir $DATASET_DIR --output-dir $OUTPUT_DIR

# show the trajectory with gt. note that use "-s" for global scale alignment
 evo_traj kitti -s $OUTPUT_DIR/$seq.txt --ref=./kitti_eval/$seq.txt -s -p --plot_mode=xz --save_plot $OUTPUT_DIR/traj.zip
 evo_ape kitti -s ./kitti_eval/$seq.txt $OUTPUT_DIR/$seq.txt -p --plot_mode=xz --save_results $OUTPUT_DIR/ate.zip
 evo_rpe kitti -s ./kitti_eval/$seq.txt $OUTPUT_DIR/$seq.txt -p --plot_mode=xz --save_results $OUTPUT_DIR/rpe.zip

#python /home/yoyee/Documents/deep_keyframe/kitti-odom-eval/eval_odom.py --result $OUTPUT_DIR --align 7dof --seqs $seq
