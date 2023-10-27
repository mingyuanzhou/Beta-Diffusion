#!/bin/bash
# usage: sh monitor.sh <folder to monitor (usually the same as the one used as --outdir for train.py)> <name of .pkl file under the folder to monitor>  <path of fid .npz file>  <number of steps in reverse diffusion> >> <target_folder/log_FID.txt>

#sh FID.sh 'betadiff-train-runs/00001-cifar10-32x32-uncond-ddpmpp-betadiff-gpus8-batch512-fp32' 'network-snapshot-200000.pkl' 'fid-refs/cifar10-32x32.npz' 10 >> betadiff-train-runs/00001-cifar10-32x32-uncond-ddpmpp-betadiff-gpus8-batch512-fp32/FID.txt

folder_to_monitor=$1 
pkl_file_name=$2
fid_file=$3
steps=$4

echo "\n\n\n The number of NFEs is '$steps'"

python -m torch.distributed.run --standalone --nproc_per_node=8 generate.py --steps=$steps --outdir=$folder_to_monitor/images --network=$folder_to_monitor/$pkl_file_name --seeds=0-49999
python -m torch.distributed.run --standalone --nproc_per_node=8 fid.py calc --images=$folder_to_monitor/images --ref=$fid_file
python -m torch.distributed.run --standalone --nproc_per_node=8 fid.py calc --images=$folder_to_monitor/images_1 --ref=$fid_file
