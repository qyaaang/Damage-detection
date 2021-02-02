data_sources=("1" "2" "3")
dim_inputs=(300 400)
for data_source in "${data_sources[@]}"; do
  for dim_input in "${dim_inputs[@]}"; do
      printf "\033[1;32mData source:\t%s\nDim input:\t%s\n\033[0m" "$data_source" "$dim_input"
      python3 data_preparation.py --dim_input "$dim_input" --data_source "$data_source"
  done
done
