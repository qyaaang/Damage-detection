sys=$(uname -a)
Mac="Darwin"
len_segs=(300 400 500)
datasets=("W-2" "W-5" "W-7")
net_names=("MLP" "Conv2D")
num_hidden_maps=(256 128 64 32)
learning_rate=1e-4
for len_seg in "${len_segs[@]}"; do
  for net_name in "${net_names[@]}"; do
    for dataset in "${datasets[@]}"; do
      if [ "$net_name" == "MLP" ]; then
          printf "\033[1;32mDataset:\t%s\nLength of segments:\t%s\nNet name:\t%s\n\033[0m" \
                 "$dataset" "$len_seg" "$net_name"
          num_epoch=10000
          if [[ $sys =~ $Mac ]]; then
              python3 test.py --dataset "$dataset" --model_name AE --net_name "$net_name" \
                              --len_seg "$len_seg" \
                              --num_epoch $num_epoch --learning_rate $learning_rate \
                              --alpha 20
          else
              python test.py --dataset "$dataset" --model_name AE --net_name "$net_name" \
                             --len_seg "$len_seg" \
                             --num_epoch $num_epoch --learning_rate $learning_rate \
                             --alpha 20
          fi
      else
          num_epoch=1000
          for num_hidden_map in "${num_hidden_maps[@]}"; do
            printf "\033[1;32mDataset:\t%s\nLength of segments:\t%s\nNet name:\t%s\nNum hidden maps:\t%s\n\033[0m" \
                   "$dataset" "$len_seg" "$net_name" "$num_hidden_map"
            if [[ $sys =~ $Mac ]]; then
               python3 test.py --dataset "$dataset" --model_name AE --net_name "$net_name" \
                               --len_seg "$len_seg" \
                               --num_epoch $num_epoch --num_hidden_map "$num_hidden_map" \
                               --learning_rate $learning_rate \
                               --alpha 100
            else
               python test.py --dataset "$dataset" --model_name AE --net_name "$net_name" \
                              --len_seg "$len_seg" \
                              --num_epoch $num_epoch --num_hidden_map "$num_hidden_map" \
                              --learning_rate $learning_rate \
                              --alpha 100
            fi
          done
      fi
    done
  done
done
