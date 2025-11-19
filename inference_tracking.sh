#!/bin/sh


#module load gcc/13.2.0
module load gcc/13.2.0-nvptx
module load ffmpeg

export OPENCV_FFMPEG_READ_ATTEMPTS=50000
for f in ../.project/dir.project/workshop-data/videos/*.MP4; do
#GH010590 GH010592 GH010593 GH010595 GH010596 GH010598 GH010599 GH010600 GH010601 GH010603 GH010604 GH010606 GH010607 GH010609 GH010612 GH010614 GH010615 GH010618 GH010625 GH010627 GH010630 GH010631 GH010632 GH010633 GH010634 GH010636 GH010637 GH010663 GH010664; do 
    vid="$(basename "${f%.MP4}")"
    for conf in 0.02; do
    for det in 0.3; do #0.4 0.5 0.6
    for sim in 0.8; do #0.7 0.8 0.9
    for seed in 1; do #2 3
    for prop in 1; do
    for buffer in 3; do

python PriMAT-tracking/src/demo.py mot  --load_tracking_model ../.project/dir.project/workshop-models/tracking_model_50.pth\
                    --conf_thres "$conf"\
                    --det_thres "$det"\
                    --new_overlap_thres 0.8\
                    --sim_thres "$sim"\
                    --input_video ../.project/dir.project/workshop-data/videos/"$vid".MP4\
                    --output_root PriMAT-tracking/videos/\
                    --output_name "$vid"\
                    --store_opt\
                    --line_thickness 2\
                    --debug_info\
                    --arch hrnet_32\
                    --output_format text\
                    --reid_cls_names monkey,snake,hand,cube,popcorn,peanut\
                    --proportion_iou "$prop"\
                    --track_buffer "$buffer"
done
done
done
done
done
done
rm -r PriMAT-tracking/videos/frame
done