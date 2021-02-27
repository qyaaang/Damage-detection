net_names=("MLP" "Conv2D")
num_hidden_maps=(256 128 64 32)
seg_no=10
for net_name in "${net_names[@]}"; do
  if [ "$net_name" == "MLP" ]; then
    for seg_idx in $(seq $seg_no $seg_no); do
      printf "\033[1;32mNet name:\t%s\nSegment No.:\t%s\n\033[0m" \
             "$net_name" "$seg_idx"
      python3 reconstruction.py --net_name "$net_name" --seg_idx "$seg_idx"
    done
  else
    for seg_idx in $(seq $seg_no $seg_no); do
      for num_hidden_map in "${num_hidden_maps[@]}"; do
        printf "\033[1;32mNet name:\t%s\nSegment No.:\t%s\nNum hidden maps:\t%s\n\033[0m" \
               "$net_name" "$seg_idx" "$num_hidden_map"
        python3 reconstruction.py --net_name "$net_name" --num_epoch 1000 \
                                  --num_hidden_map "$num_hidden_map" --seg_idx "$seg_idx"
      done
    done
  fi
done
