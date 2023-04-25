#!/bin/bash
mkdir -p results
mkdir -p results/log
mkdir -p generated_datasets
mkdir -p err_dist_files

if [[ -d "python_venv" ]]; then
  . ./python_venv/bin/activate
fi

if [ "$HOSTNAME" = "tango" ]; then
#  files=("beers hospital movies rayyan flights tax")
#  scales=("2 4 4.5 8 16 32 32.5 64 128 256 256.5 512 1024")
  scales=("128 256")
  files=("tax beers hospital movies rayyan flights")
  scalesFraction=10
elif [ "$HOSTNAME" = "india" ]; then
  scales=("128 256")
  files=("beers hospital movies rayyan flights tax")
  scalesFraction=10
else
#  files=("rayyan flights")
  files=("tax beers hospital movies rayyan flights")
  scales=("2 16")
  scalesFraction=2
fi


for scale in $scales; do
  for item in $files; do
    log_file="./results/log/${item}_${scale}_${HOSTNAME}.log"
    ( time python3 ./test_scale_up_local.py \
      -c datasets/clean_${item}.csv -d datasets/dirty_${item}.csv \
      -sf $scale -edc err_dist_files/local_original_dist_${item}.txt \
      -edd err_dist_files/local_dirty_dist_${item}_${scale}.txt  \
      -o generated_datasets/generated_${item}_${scale}.csv; ) \
      > $log_file 2>&1

      # Delete generated file
      rm -f generated_datasets/generated_${item}_${scale}.csv
  done
done
#  # Scaling fractions
#  for scale in $(seq $scalesFraction); do
#    for fraction in $(seq $scalesFraction); do
#      log_file="./results/log/${item}_${scale}.${fraction}_${HOSTNAME}.log"
#      ( time python3 ./test_scale_up_local.py \
#        -c datasets/clean_${item}.csv -d datasets/dirty_${item}.csv \
#        -sf ${scale}.${fraction} -edc err_dist_files/local_original_dist_${item}.txt \
#        -edd err_dist_files/local_dirty_dist_${item}_${scale}.${fraction}.txt  \
#        -o generated_datasets/generated_${item}_${scale}.${fraction}.csv; ) \
#        > $log_file 2>&1
#
##        # Delete generated file
##        rm -f generated_datasets/generated_${item}_${scale}.${fraction}.csv
#      done
#  done
wait
