import gc
import random
import time
import typing

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

import compare_error_dist
import static_utils
import utils
from error_distribution import ErrorDistribution
from error_sequence_generator import generate_errors, generate_errors_sp
from scale_dataset import DataGen, DataGenSP
from utils import Utils

random.seed(3215)


def start_spark(app_name: str, in_cluster: bool = False):
    if in_cluster:
        # TODO heartbeatInterval timeout 1000s
        spark = SparkSession.builder\
            .master("yarn") \
            .appName(app_name) \
            .config("spark.driver.memory", "90g") \
            .config("spark.executor.memory", "100g") \
            .config("spark.driver.cores", "32")\
            .config("spark.submit.deployMode", "client") \
            .config("spark.executor.heartbeatInterval", "50s") \
            .config("spark.network.timeout", "100000s") \
            .config("spark.executor.instances", "3") \
            .getOrCreate()

    else:
        spark = SparkSession.builder.master("local[8]") \
            .appName(app_name) \
            .config("spark.driver.memory", "9g").getOrCreate()

    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark.conf.set("spark.sql.codegen.wholeStage", "false")
    spark.conf.set("spark.sql.shuffle.partitions", "6")

    spark.sparkContext.setLogLevel("ERROR")

    return spark


def read_dataset(data_path: str, has_header: int, sep: str = ',', schema: typing.Dict = {}) -> pd.DataFrame:
    if len(schema) > 0:
        return pd.read_csv(data_path, header=has_header, sep=sep, dtype=schema, engine='pyarrow')
    else:
        return pd.read_csv(data_path, header=has_header, sep=sep, engine='pyarrow')


def run_default(clean_path: str, dirty_path: str, scaling_factor: float,
                clean_header: int = 0, dirty_header: int = 0,
                clean_sep: str = ',', dirty_sep: str = ',', fds_file_path: str = None,
                original_dist_file: str = 'err_dist_files/original_dist.txt',
                gen_dist_file: str = 'err_dist_files/scale_dist.txt',
                output_path: str = 'generated_datasets/gen_data.csv'):
    # Read clean dataset
    clean_data = read_dataset(clean_path, clean_header, clean_sep)
    # Just in case, because not always clean dataset is without MV, e.g. taxes
    # clean_data.mask(clean_data.isnull(), inplace=True)
    clean_data.fillna(0, inplace=True)

    # Infer schema
    schema, schema_safe = Utils.detect_schema_from_rows_sample(clean_data)

    # Read dirty dataset
    dirty_data = read_dataset(dirty_path, dirty_header, dirty_sep)

    # Return dirty dataset or its slice if scaling_factor <=1
    if scaling_factor == 1:
        print("Scaling factor is 1: return dirty data without generation.")
        return dirty_data

    elif scaling_factor <= 1:
        print(f"Scaling factor is {scaling_factor}: return slice of dirty data without generation.")
        return dirty_data.loc[1:int(scaling_factor * dirty_data.shape[0])]

    start = time.time()

    # Compute error distributions
    err_dist = ErrorDistribution(schema=schema, rows=dirty_data.shape[0], cols=dirty_data.shape[1])
    err_dist.get_error_dist(clean_data=clean_data, dirty_data=dirty_data,
                            scaling_factor=scaling_factor, fds_path=fds_file_path)
    dist_done = time.time()
    print(f"Error distribution computation: {dist_done - start}\n")

    # Scale clean dataset
    data_gen = DataGen()

    data_gen.generate(data=clean_data, error_distribution=err_dist, scaling_factor=scaling_factor,
                      new_dim=err_dist.generated_rows, dirty_schema=dirty_data.dtypes.to_dict())

    scale_done = time.time()
    print(f"Scale dataset: {scale_done - dist_done}\n")

    # Generate error sequence and apply it
    data_gen.clean_data_scaled, new_error_dist = generate_errors(scaled_data=data_gen.clean_data_scaled,
                                                                 scaled_schema=schema, err_dist=err_dist,
                                                                 scaling_factor=scaling_factor)
    errors_done = time.time()
    print("Generated dataset")
    print(f"Errors to scaled dataset: {errors_done - scale_done}\n")

    save_dataset = time.time()
    data_gen.clean_data_scaled.to_csv(index=False, path_or_buf=output_path)
    print(f"Save dataset: {time.time() - save_dataset}\n")

    dist_after_start = time.time()
    new_error_dist.get_error_dist_after_scaling(data_gen.clean_data_scaled)
    dist_after_done = time.time()
    print(f"Computed new error distribution: {dist_after_done - dist_after_start}\n")

    utils.write_err_dist(err_dist, original_dist_file, False)
    utils.write_err_dist(new_error_dist, gen_dist_file, False)

    valid_begin = time.time()

    # Validate (soft constraints)
    compare_error_dist.compare_error_distributions_statistics(cols=data_gen.clean_data_scaled.columns,
                                                              original=err_dist, generated=new_error_dist)

    valid_done = time.time()

    print(f"Validation of scaled dataset: {valid_done - valid_begin}\n")
    print(f"Time elapsed {valid_done-start}\n")

    return data_gen.clean_data_scaled


def run_distributed(clean_path: str, dirty_path: str, scaling_factor: float, output_path: str,
                    clean_header: int = 0, dirty_header: int = 0,
                    clean_sep: str = ',', dirty_sep: str = ',', fds_file_path: str = None,
                    original_dist_file: str = 'err_dist_files/original_dist.txt',
                    gen_dist_file: str = 'err_dist_files/scale_dist.txt'):
    # Read clean dataset
    clean_data = read_dataset(clean_path, clean_header, clean_sep)
    # Just in case, because not always clean dataset is without MV, e.g. taxes
    # clean_data.mask(clean_data.isnull(), inplace=True)
    clean_data.fillna(0, inplace=True)

    # Infer schema
    schema, schema_safe = Utils.detect_schema_from_rows_sample(clean_data)

    # Read dirty dataset
    dirty_data = read_dataset(dirty_path, dirty_header, dirty_sep)

    # Return dirty dataset or its slice if scaling_factor <=1
    if scaling_factor == 1:
        print("Scaling factor is 1: return dirty data without generation.")
        return dirty_data

    elif scaling_factor <= 1:
        print(f"Scaling factor is {scaling_factor}: return slice of dirty data without generation.")
        return dirty_data.loc[1:int(scaling_factor * dirty_data.shape[0])]

    start = time.time()

    # Compute error distributions - locally
    err_dist = ErrorDistribution(schema=schema, rows=dirty_data.shape[0], cols=dirty_data.shape[1])
    err_dist.get_error_dist(clean_data=clean_data, dirty_data=dirty_data,
                            scaling_factor=scaling_factor, fds_path=fds_file_path)
    dist_done = time.time()

    # Create spark session
    file_name = dirty_path.split("//")[-1].replace("//", "").replace("datasets/", "").replace(".csv", "")
    spark = start_spark(app_name=f"DataGenerator_{file_name}_{scaling_factor}", in_cluster=static_utils.is_in_cluster())

    data_gen = DataGenSP()
    data_gen.generate(clean_df=clean_data, spark=spark, data_path=clean_path, dirty_path=dirty_path,
                      scaling_factor=scaling_factor, new_dim=err_dist.generated_rows,
                      rows=clean_data.shape[0], cols=clean_data.shape[1])

    scale_done = time.time()

    data_gen.clean_data_scaled, new_error_dist = generate_errors_sp(spark=spark, generator=data_gen, scaled_schema=schema,
                                                                    err_dist=err_dist, scaling_factor=scaling_factor)
    errors_done = time.time()

    print("Generated dataset")
    save_dataset = time.time()

    # Write to the file
    # data_gen.clean_data_scaled.write.format("csv").mode("overwrite").save(output_path)
    data_gen.clean_data_scaled.write.format("parquet").mode("overwrite").option("path", output_path).saveAsTable("dirty")

    print(f"Save dataset: {time.time() - save_dataset}\n")

    dist_after_start = time.time()

    new_error_dist.get_error_dist_after_scaling_sp(data_gen.clean_data_scaled, dirty_data.dtypes)
    dist_after_done = time.time()

    print(f"Error distribution computation: {dist_done - start}\n")
    print(f"Scale dataset: {scale_done - dist_done}\n")
    print(f"Errors to scaled dataset: {errors_done - scale_done}\n")
    print(f"Generated error distribution: {dist_after_done - dist_after_start}\n")

    utils.write_err_dist(err_dist, original_dist_file, True)
    utils.write_err_dist(new_error_dist, gen_dist_file, True)

    valid_begin = time.time()
    compare_error_dist.compare_error_distributions_statistics_sp(
        cols=data_gen.clean_data_scaled.columns, original=err_dist, generated=new_error_dist)
    valid_done = time.time()
    print(f"Validation {valid_done - valid_begin}\n")
    print(f"Time elapsed {valid_done - start}\n")

    return data_gen.clean_data_scaled
