#!/bin/bash

set -x

# for i in 0{1..9} {10..24} ; do
# #for i in 0{1..2} ; do
# 	optirun ./image_stab lowres/lowres_img01.exr lowres/lowres_img$i.exr
# 	exrtopng t_lowres/lowres_img$i.exr t_lowres/lowres_img$i.png
# done

time optirun ./image_stab lowres/lowres_img 24

for i in 0{1..9} {10..24}; do
	exrtopng t_lowres/lowres_img$i.exr t_lowres/lowres_img$i.png > /dev/null
done

ffmpeg -r 20 -i t_lowres/lowres_img%02d.png t_lowres/output.mp4
