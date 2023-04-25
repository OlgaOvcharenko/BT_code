#!/bin/bash

mkdir -p results
mkdir -p results/log
mkdir -p generated_datasets
mkdir -p err_dist_files

# Load env
if [[ -d "python_venv" ]]; then
  . ./python_venv/bin/activate
fi

if [ "$HOSTNAME" = "tango" ]; then
#  files=("beers hospital movies rayyan flights tax")
#  scales=("2 8 64 256 2048 8192 32768 65536")

#  files=("rayyan")
#  scales=("65536")


  files=("movies rayyan flights")
  scales=("8192")

  source scripts/load-had-3.3-spark-3.2-java-11.sh

  hdfs dfs -rm -r tmp
  hdfs dfs -mkdir tmp

else
#  files=("tax beers hospital movies rayyan flights")
  files=("movies flights")
  scales=("4 8")
  scalesFraction=2
fi

for scale in $scales; do
  for item in $files; do
    log_file="./results/log/${item}_${scale}_${HOSTNAME}_distributed.log"

    hdfs dfs -rm err_dist_files/distributed_original_dist_${item}_${scale}.txt
    hdfs dfs -rm err_dist_files/distributed_dirty_dist_${item}_${scale}.txt
    hdfs dfs -rm -r generated_datasets/generated_distributed_${item}_${scale}.parquet

    ( time spark-submit \
      --master yarn \
      --deploy-mode client \
      --num-executors 3 \
      --driver-memory 90g \
      --conf spark.driver.extraJavaOptions="-Xms90g -Xmn9g" \
      --conf spark.executor.heartbeatInterval=50s \
      --conf spark.network.timeout=100000s \
      --conf spark.executor.instances=3 \
      ./test_scale_up_distributed.py \
      -c datasets/clean_${item}.csv \
      -d datasets/dirty_${item}.csv \
      -sf $scale -edc err_dist_files/distributed_original_dist_${item}_${scale}.txt \
      -edd err_dist_files/distributed_dirty_dist_${item}_${scale}.txt  \
      -o generated_datasets/generated_distributed_${item}_${scale}.parquet; ) \
      > $log_file 2>&1

      # Delete generated file
#      rm -f generated_datasets/generated_${item}_${scale}.csv

      if [ "$HOSTNAME" = "tango" ]; then
        hdfs dfs -rm -r tmp
        hdfs dfs -mkdir tmp
      fi
  done
done
wait
