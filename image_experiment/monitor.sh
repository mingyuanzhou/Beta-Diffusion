#!/bin/bash
# Generate 1000 images to calcuate FID for each new pkl file
# usage: sh monitor.sh <folder to monitor (usually the same as the one used as --outdir for train.py)> <path of fid .npz file> >> <target_folder/log_FID.txt>
folder_to_monitor=$1
fid_file=$2
steps=$3

latest_snapshot=""
last_update=$(date +%s)

timeout_seconds=$((24 * 60 * 60))  # 24 hours in seconds


while true; do
    newest_file=$(ls -t "$folder_to_monitor"/network-snapshot-*.pkl 2>/dev/null | head -n1)
    if echo "$newest_file" | grep -q "network-snapshot-[0-9]\+\.pkl"; then
        snapshot_number=$(echo "$newest_file" | sed -n -e 's/^.*network-snapshot-\([0-9]\+\)\.pkl$/\1/p')
        if [ -z "$snapshot_number" ] || ! echo "$snapshot_number" | grep -q '^[0-9]\+$'; then
            continue
        fi
        #if [ "$snapshot_number" -gt 0 ] && [ -z "$latest_snapshot" ] || [ "$snapshot_number" -gt "$latest_snapshot" ]; then
        if [ "$snapshot_number" -gt 0 ] && { [ -z "$latest_snapshot" ] || [ "$snapshot_number" -gt "$latest_snapshot" ]; }; then

            latest_snapshot=$snapshot_number
            echo "\n\n\n The newest file is '$newest_file'"
            python -m torch.distributed.run --standalone --nproc_per_node=8 generate_monitor.py --steps=$steps --outdir=$folder_to_monitor/images --network=$newest_file --seeds=0-999
            python -m torch.distributed.run --standalone --nproc_per_node=8 fid.py calc --images=$folder_to_monitor/images --ref=$fid_file --num=1000
            python -m torch.distributed.run --standalone --nproc_per_node=8 fid.py calc --images=$folder_to_monitor/images_1 --ref=$fid_file --num=1000
            last_update=$(date +%s)
            
        fi
    fi
    current_time=$(date +%s)
    if [ $((current_time - last_update)) -gt $timeout_seconds ]; then
        echo "No new files in the last 24 hours, stopping."
        break
    fi
    sleep 0.02h
done
