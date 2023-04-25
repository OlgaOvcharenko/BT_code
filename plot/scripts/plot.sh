#!/bin/bash
# files=("tax beers hospital movies rayyan flights")
# for item in $files; do
#     python3 ./plot/scripts/distinct.py -f err_dist_files/local_gen_dist_$item.txt -o plot/distinct/${item}_distinct_2x/ &
#     python3 ./plot/scripts/distinct.py -f err_dist_files/local_original_dist_$item.txt -o plot/distinct/${item}_distinct/ &
# done


#python3 ./plot/scripts/stackedErrors.py \
#   -f err_dist_files/local_original_dist_ \
#   -d tax beers flights hospital movies rayyan \
#   -o plot/errors &


#python3 ./plot/scripts/statistics.py \
#    -d results/log/tango_local \
#    -n tax \
#    -l True \
#    -o plot/beers &

python3 ./plot/scripts/max_p_diff.py \
    -d results/log/tango_local \
    -n beers \
    -l True \
    -o plot/beers &

#
#python3 ./plot/scripts/statistics.py \
#    -d tango_dist \
#    -n flights \
#    -l False \
#    -o plot/beers &
#
#python3 ./plot/scripts/statistics.py \
#    -d tango_dist \
#    -n hospital \
#    -l False \
#    -o plot/beers &
#
#python3 ./plot/scripts/statistics.py \
#    -d tango_dist \
#    -n movies \
#    -l False \
#    -o plot/beers &
#
#
#python3 ./plot/scripts/statistics.py \
#    -d tango_dist \
#    -n ryyan \
#    -l False \
#    -o plot/beers &




wait
