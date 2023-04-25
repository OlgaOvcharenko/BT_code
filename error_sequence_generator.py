import copy
import math
import multiprocessing
import random
import time
import typing
from multiprocessing import Pool

import pandas as pd
import pyspark.sql.functions
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

import error_distribution
import static_utils
import utils
from error_distribution import ErrorDistribution
from errors import AddTypos, AddMissingValues, AddOutliers, AddReplacements, AddSwaps
from scale_dataset import DataGenSP


def _log_results(log_str: typing.List[str]):
    if log_str:
        print(*log_str, sep="\n")


def generate_errors_sp(spark: SparkSession, generator: DataGenSP, scaled_schema: typing.Dict,
                       err_dist: ErrorDistribution, scaling_factor: float):

    new_error_dist = error_distribution.ErrorDistribution(schema=scaled_schema,
                                                          rows=generator.num_rows, cols=generator.num_cols)
    new_error_dist.get_scaled_stats_with_errors(scaling_factor=scaling_factor)

    # generator.clean_data_scaled.filter(pyspark.sql.functions.col("article_jcreated_at") == "1/4/01").groupBy(
    #     "article_jcreated_at").count().show(n=10000)
    # exit()

    col_dist_counts = dict()
    col_i = 0
    col_names = generator.clean_data_scaled.columns
    col_names.remove(generator.index_col_name)

    replacements_indices = dict()
    for col in col_names:
        # Replacements
        replacements = round(err_dist.distinct_replacements_count[col] * scaling_factor)
        if replacements > 0:
            print(f"Replacements {col}")

            replacements_indices[col] = _add_replacements_distributed(
                spark=spark, generator=generator, err_dist=err_dist, scaling_factor=scaling_factor, col_name=col,
                new_err_dist=new_error_dist, replacements=replacements)

            # _add_replacements_distributed_join(
            #     generator=generator, err_dist=err_dist, scaling_factor=scaling_factor, col=col,
            #     new_err_dist=new_error_dist, replacements=replacements)

    for col in col_names:
        num_outliers = round(err_dist.outliers_num[col] * scaling_factor)
        if num_outliers > 0:
            new_error_dist.dirty_mean[col] = list(generator.clean_data_scaled
                                                  .select(pyspark.sql.functions.mean(col))
                                                  .collect()[0].asDict().values())[0] \
                if int(scaling_factor) != scaling_factor else err_dist.dirty_mean[col]

        new_error_dist.outliers_num[col] = num_outliers

    pool = Pool(processes=multiprocessing.cpu_count())  # multiprocessing.cpu_count()
    tasks = list()
    n_r = copy.deepcopy(generator.num_rows)
    index_name = copy.deepcopy(generator.index_col_name)
    for col in col_names:
        kwargs = {"num_rows": n_r, "index_col_name": index_name,
                  "col_dist_counts": col_dist_counts, "col": col, "col_i": col_i,
                  "err_dist": err_dist, "new_error_dist": new_error_dist,
                  "scaling_factor": scaling_factor,
                  "replacements_indices": replacements_indices.get(col, [])}

        tasks.append(
            pool.apply_async(_run_col_distributed, (), kwargs))

        col_i += 1

    for t in tasks:
        results = t.get()
        if results:
            _log_results(results[0])

            col_tmp = results[2]
            i_col_tmp = col_names.index(col_tmp)

            tmp_err_dist = results[1]
            new_error_dist.typos_values[col_tmp] = tmp_err_dist.typos_values.get(col_tmp, dict())
            new_error_dist.typos_num[col_tmp] = tmp_err_dist.typos_num.get(col_tmp, 0)
            new_error_dist.mv_values[col_tmp] = tmp_err_dist.mv_values.get(col_tmp, dict())
            new_error_dist.mv_num[col_tmp] = tmp_err_dist.mv_num.get(col_tmp, 0)
            new_error_dist.distinct_replacements_count[col_tmp] = tmp_err_dist.distinct_replacements_count.get(
                col_tmp, 0)
            new_error_dist.replacements_dicts[col_tmp] = tmp_err_dist.replacements_dicts.get(col_tmp, dict())
            new_error_dist.initial_difference_mask[:, i_col_tmp] = tmp_err_dist.initial_difference_mask[:, i_col_tmp]
            new_error_dist.mv_mask[:, i_col_tmp] = tmp_err_dist.mv_mask[:, i_col_tmp]
            new_error_dist.outliers_num[col_tmp] = tmp_err_dist.outliers_num.get(col_tmp, 0)
            new_error_dist.outliers_values[col_tmp] = tmp_err_dist.outliers_values.get(col_tmp, 0)

    pool.close()
    pool.join()

    log_str = list()
    col_i = 0
    for col in col_names:
        time_set = time.time()
        if err_dist.num_errors_per_col[col_i] == 0:
            col_i += 1
            continue

        col_modified_name = col + "_modified"
        errors_tmp_path = "tmp/" + col_modified_name + "_tmp"
        cols = [generator.index_col_name, col_modified_name]

        new_data_frame = spark.read.csv(errors_tmp_path).toDF(*cols)

        log_str.append(f"Create error DATAFRAME  {time.time() - time_set}")

        new_cols = [c for c in generator.clean_data_scaled.columns if c not in [col, col_modified_name]]
        new_cols.append(F.when(new_data_frame[col_modified_name].isNull(), generator.clean_data_scaled[col])
                        .otherwise(new_data_frame[col_modified_name]).alias(col))

        log_str.append(f"Create spark obj {time.time() - time_set}")

        generator.clean_data_scaled = generator.clean_data_scaled \
            .join(new_data_frame, generator.index_col_name, "left_outer") \
            .select(new_cols)

        log_str.append(f"Set values after join, TOTAL {time.time() - time_set}")

        col_i += 1

    print(*log_str, sep="\n")

    # Swaps numerical
    print("Swaps numerical done")
    num_swaps = sum([sum(d.values()) for d in err_dist.swaps_dict_nums.values()]) * scaling_factor
    if num_swaps > 0 and len(err_dist.swaps_dict_nums) > 0:
        # generator.clean_data_scaled, new_error_dist.swaps_dict_nums =
        _add_numerical_swaps(spark=spark, generator=generator, err_dist=err_dist, scaling_factor=scaling_factor,
                             new_err_dist=new_error_dist, num_swaps=num_swaps)

    # Swap str-str, str-numeric
    print("Swaps str")
    num_swaps = sum([sum(d.values()) for d in err_dist.swaps_dict_with_str.values()]) * scaling_factor
    if num_swaps > 0 and len(err_dist.swaps_dict_nums) > 0:
        _add_str_swaps(spark=spark, generator=generator, err_dist=err_dist, scaling_factor=scaling_factor,
                       new_err_dist=new_error_dist, num_swaps=num_swaps)

    generator.clean_data_scaled = generator.clean_data_scaled\
        .repartition(DataGenSP.get_number_partitions(rows=generator.num_rows, cols=generator.num_cols), generator.index_col_name)\
        .drop(generator.index_col_name)\
        .cache()

    return generator.clean_data_scaled, new_error_dist


def _add_numerical_swaps(
        spark: SparkSession, generator: DataGenSP, err_dist: ErrorDistribution, scaling_factor: float,
        new_err_dist: ErrorDistribution, num_swaps: int):
    swap_start = time.time()
    swaps_paths, tmp_swaps, col_names_modified = AddSwaps().run_sp(
        scaled_data=generator.clean_data_scaled, num_swaps=num_swaps, scaling_factor=scaling_factor,
        swaps_dict=err_dist.swaps_dict_nums, index_col=generator.index_col_name, original_rows=err_dist.original_rows)

    for cols_original, path, col_names, num_swaps_pair in \
            zip(err_dist.swaps_dict_nums.keys(), swaps_paths, col_names_modified, tmp_swaps):
        cols = [generator.index_col_name, col_names[0], col_names[1]]
        new_data_frame = spark.read.csv(path).toDF(*cols)

        print(f"Create SWAP DATAFRAME  {time.time() - swap_start}")

        new_cols = [c for c in generator.clean_data_scaled.columns if c not in [cols_original[0], cols_original[1], col_names[0], col_names[1]]]
        new_cols.append(F.when(new_data_frame[col_names[0]].isNull(), generator.clean_data_scaled[cols_original[0]])
                        .otherwise(new_data_frame[col_names[0]]).alias(cols_original[0]))
        new_cols.append(F.when(new_data_frame[col_names[1]].isNull(), generator.clean_data_scaled[cols_original[1]])
                        .otherwise(new_data_frame[col_names[1]]).alias(cols_original[1]))

        print(f"SWAP function new column {time.time() - swap_start}")

        generator.clean_data_scaled = generator.clean_data_scaled \
            .join(new_data_frame, generator.index_col_name, "left_outer") \
            .select(new_cols)

        print(f"SWAP Create spark obj {time.time() - swap_start}")

        new_err_dist.swaps_dict_nums[cols_original] = {"1": num_swaps_pair}
    # return generator.clean_data_scaled, new_err_dist.swaps_dict_nums


def _add_str_swaps(
        spark: SparkSession, generator: DataGenSP, err_dist: ErrorDistribution, scaling_factor: float,
        new_err_dist: ErrorDistribution, num_swaps: int):
    swap_start = time.time()
    swaps_paths, tmp_swaps, col_names_modified = AddSwaps().run_sp(
        scaled_data=generator.clean_data_scaled, num_swaps=num_swaps, scaling_factor=scaling_factor,
        swaps_dict=err_dist.swaps_dict_with_str, index_col=generator.index_col_name, original_rows=err_dist.original_rows)

    for cols_original, path, col_names, num_swaps_pair in \
            zip(err_dist.swaps_dict_with_str.keys(), swaps_paths, col_names_modified, tmp_swaps):
        cols = [generator.index_col_name, col_names[0], col_names[1]]
        new_data_frame = spark.read.csv(path).toDF(*cols)

        print(f"Create SWAP DATAFRAME  {time.time() - swap_start}")

        new_cols = [c for c in generator.clean_data_scaled.columns if c not in [cols_original[0], cols_original[1], col_names[0], col_names[1]]]
        new_cols.append(F.when(new_data_frame[col_names[0]].isNull(), generator.clean_data_scaled[cols_original[0]])
                        .otherwise(new_data_frame[col_names[0]]).alias(cols_original[0]))
        new_cols.append(F.when(new_data_frame[col_names[1]].isNull(), generator.clean_data_scaled[cols_original[1]])
                        .otherwise(new_data_frame[col_names[1]]).alias(cols_original[1]))

        print(f"SWAP function new column {time.time() - swap_start}")

        generator.clean_data_scaled = generator.clean_data_scaled \
            .join(new_data_frame, generator.index_col_name, "left_outer") \
            .select(new_cols)

        print(f"SWAP Create spark obj {time.time() - swap_start}")

        new_err_dist.swaps_dict_with_str[cols_original] = {"1": num_swaps_pair}


def _add_replacements_distributed(
        spark: SparkSession, generator: DataGenSP, err_dist: ErrorDistribution, scaling_factor: float,
        col_name: str, new_err_dist: ErrorDistribution, replacements: int):

    replacements_start = time.time()
    print("REPLACEMENT start")
    print(f"REPLACEMENT Num repl \t{replacements}")
    repl_tmp_path, count_replacements, col_name_modified, replacements_ind_list = AddReplacements() \
        .run_sp(scaled_column=generator.clean_data_scaled
                .select(pyspark.sql.functions.col(generator.index_col_name), pyspark.sql.functions.col(col_name)),
                col_name=col_name, num_replacements=replacements,
                scaling_factor=scaling_factor, replacements_dict=err_dist.replacements_dicts[col_name],
                err_dist=err_dist, index_col=generator.index_col_name)

    new_err_dist.distinct_replacements_count[col_name] = count_replacements

    cols = [generator.index_col_name, col_name_modified]
    new_data_frame = spark.read.csv(repl_tmp_path).toDF(*cols)

    print(f"REPLACEMENT Create replacement DATAFRAME  {time.time() - replacements_start}")

    new_cols = [c for c in generator.clean_data_scaled.columns if c not in [col_name, col_name_modified]]
    new_cols.append(F.when(new_data_frame[col_name_modified].isNull(), generator.clean_data_scaled[col_name])
                    .otherwise(new_data_frame[col_name_modified]).alias(col_name))

    print(f"REPLACEMENT Create spark obj {time.time() - replacements_start}")

    generator.clean_data_scaled = generator.clean_data_scaled \
        .join(new_data_frame, generator.index_col_name, "left_outer") \
        .select(new_cols)

    print(f"REPLACEMENT all {time.time() - replacements_start}")

    return replacements_ind_list


def _add_replacements_distributed_join(
        generator: DataGenSP, err_dist: ErrorDistribution, scaling_factor: float,
        col: str, new_err_dist: ErrorDistribution, replacements: int):

    replacements_start = time.time()
    print("Replacements start")
    print(f"Num repl \t{replacements}")
    all_changes_df, count_replacements, col_modified_name = AddReplacements() \
        .run_sp_joins(scaled_column=generator.clean_data_scaled
                      .select(pyspark.sql.functions.col(generator.index_col_name), pyspark.sql.functions.col(col)),
                      col_name=col, num_replacements=replacements,
                      scaling_factor=scaling_factor, replacements_dict=err_dist.replacements_dicts[col],
                      err_dist=err_dist, index_col=generator.index_col_name)

    new_err_dist.distinct_replacements_count[col] = count_replacements

    start_join_big_time = time.time()

    # Update column
    new_cols = [c for c in generator.clean_data_scaled.columns if c not in [col, col_modified_name]]
    all_changes_df = all_changes_df.withColumnRenamed(col, col_modified_name)
    append_val = F.when(all_changes_df[col_modified_name].isNull(), generator.clean_data_scaled[col]) \
        .otherwise(all_changes_df[col_modified_name]).alias(col)
    new_cols.append(append_val)

    generator.clean_data_scaled = generator.clean_data_scaled \
        .join(all_changes_df, generator.index_col_name, "left_outer") \
        .select(new_cols) \

    print(f"Replacements join {time.time() - start_join_big_time}")

    print(f"Replacement all {time.time() - replacements_start}")


def _run_col_distributed(num_rows: int, index_col_name: str, col_dist_counts: typing.Dict, col: str, col_i: int,
                         err_dist: error_distribution.ErrorDistribution,
                         new_error_dist: error_distribution.ErrorDistribution, scaling_factor: float,
                         replacements_indices: typing.List):
    log_str = [f"\n{str(col).upper()}"]

    # No errors to introduce n the column
    if err_dist.num_errors_per_col[col_i] == 0:
        return  # Early abort

    # Distincts
    dist_dirty_num = err_dist.distinct_num[col]
    dist_clean_num = len(err_dist.distinct_values_clean[col])
    dist_estimate = err_dist.distincts_estimate[col]
    ratio = math.ceil(dist_clean_num / dist_dirty_num)

    # Check distints -> how many new values can be added +- ratio between clean/dirty
    ratio_add_distincts = round(err_dist.original_rows * scaling_factor) * ratio
    distincts_new_num = min(dist_estimate, ratio_add_distincts) if dist_estimate > 0 else ratio_add_distincts

    col_dist_counts[col] = (distincts_new_num, dist_dirty_num)
    # Errors and counts
    new_errors = dict()

    # Typos
    num_typos = round(err_dist.typos_num[col] * scaling_factor)  # overall num typos to introduce
    if num_typos > 0:
        num_new_typos = err_dist.distinct_typos_estimate[col]

        if len(err_dist.typos_values[col]) > num_new_typos:
            num_new_typos = len(err_dist.typos_values[col])

        new_error_dist.typos_values[col] = \
            AddTypos().run_sp(col_name=col, num_typos=num_typos,
                              num_new_dist_typos=num_new_typos, old_num_typos=err_dist.typos_num[col],
                              existing_typos=err_dist.typos_values[col],
                              distincts=list(err_dist.distinct_values_clean[col].keys()))
        new_errors.update(new_error_dist.typos_values[col])

    new_error_dist.typos_num[col] = sum(new_errors.values())

    # Missing values
    num_mv = round(err_dist.mv_num[col] * scaling_factor)
    if num_mv > 0:
        # number of new distinct mv
        num_new_mv = err_dist.distinct_mv_estimate[col]

        if len(err_dist.mv_values[col]) > num_new_mv:
            num_new_mv = len(err_dist.mv_values[col])

        new_error_dist.mv_values[col], less_unique_values_then_needed = AddMissingValues().run(
            col_name=col, num_mv=num_mv, num_new_dist_mv=num_new_mv, old_num_mv=err_dist.mv_num[col],
            existing_mv=err_dist.mv_values[col])
        new_errors.update(new_error_dist.mv_values[col])
        new_error_dist.mv_num[col] = sum(new_error_dist.mv_values[col].values())
        # print(f"num mv {num_mv}")
        # print(err_dist.mv_num[col])
        # print(new_error_dist.mv_values[col])

    else:
        new_error_dist.mv_num[col] = num_mv

    # Outliers
    num_outliers = round(err_dist.outliers_num[col] * scaling_factor)
    if num_outliers > 0 and len(err_dist.outliers_values[col]) > 0:
        num_new_outliers = round(
            (len(err_dist.outliers_values[col]) * col_dist_counts[col][0]) / col_dist_counts[col][1])

        new_outliers_dict = AddOutliers().run(col_name=col, num_outliers=num_outliers,
                                              num_new_dist_outliers=num_new_outliers, err_dist=err_dist,
                                              mean_scaled=new_error_dist.dirty_mean[col],
                                              new_num_rows=num_rows)
        new_error_dist.outliers_values[col] = new_outliers_dict
        new_errors.update(new_outliers_dict)

        new_error_dist.outliers_num[col] = sum(new_outliers_dict.values())

    else:
        new_error_dist.outliers_num[col] = num_outliers

    log_str.append(f"Set values")
    time_set = time.time()

    # For random sample to introduce
    num_errors_to_introduce = new_error_dist.typos_num[col] + new_error_dist.mv_num[col] + new_error_dist.outliers_num[col]

    col_modified_name = col + "_modified"
    cols = [index_col_name, col_modified_name]

    indices = utils.rand_except(nrows=num_rows, sample_size=num_errors_to_introduce, except_list=replacements_indices)

    log_str.append(f"Random sample {time.time() - time_set}")

    errors_tmp_path = "tmp/" + col_modified_name + "_tmp"
    # print(indices)
    # print(new_errors)
    #
    # print(len(indices))
    # print(len(new_errors.values()))

    t = 0
    if static_utils.is_in_cluster():
        hdfs_client = static_utils.write_to_hdfs()
        with hdfs_client.write(errors_tmp_path, encoding="utf-8") as file_errors:

            csv_buf = f"{cols[0]},{cols[1]}\n"
            for error_count in new_errors.items():
                for j in range(error_count[1]):
                    csv_str = f"{indices[t]},{error_count[0]}\n"
                    csv_buf += csv_str

                    if len(csv_buf) > 2000000:
                        file_errors.write(csv_buf)
                        csv_buf = ""

                    t += 1

            if len(csv_buf) > 0:
                file_errors.write(csv_buf)
                csv_buf = ""

    else:
        with open(errors_tmp_path, "w") as file_errors:

            csv_buf = f"{cols[0]},{cols[1]}\n"

            for error_count in new_errors.items():
                for j in range(error_count[1]):
                    csv_str = f"{indices[t]},{error_count[0]}\n"
                    csv_buf += csv_str

                    if len(csv_buf) > 2000000:
                        file_errors.write(csv_buf)
                        csv_buf = ""

                    t += 1

            if len(csv_buf) > 0:
                file_errors.write(csv_buf)
                csv_buf = ""

    log_str.append(f"Create errors lists sample {time.time() - time_set}")

    return log_str, new_error_dist, col


def _log_returns_local(results: typing.Tuple):
    if results:
        for k_v in results[0].items():
            results[1][k_v[0]] = k_v[1]

        # results[1].update(results[0])
        results[3].extend(results[2])


def generate_errors(scaled_data: pd.DataFrame, scaled_schema: typing.Dict,
                    err_dist: ErrorDistribution, scaling_factor: float):

    new_error_dist = error_distribution.ErrorDistribution(
        schema=scaled_schema, rows=scaled_data.shape[0], cols=scaled_data.shape[1])
    new_error_dist.get_scaled_stats_with_errors(scaling_factor=scaling_factor)

    col_dist_counts_global = dict()
    cols_outliers_global = list()

    pool = Pool(processes=multiprocessing.cpu_count())  # multiprocessing.cpu_count()
    tasks = list()

    col_i = 0
    for col in scaled_data.columns:
        kwargs = {"scaled_data_tmp_col": scaled_data[col], "scaled_schema": scaled_schema,
                  "err_dist": err_dist, "col": col, "i": col_i,
                  "new_error_dist": new_error_dist, "scaling_factor": scaling_factor}

        tasks.append(
            pool.apply_async(_get_local_errors_per_col, (), kwargs))  # , callback=_log_returns_local

        col_i += 1

    for t in tasks:
        results = t.get()

        if results:
            col_tmp = results[3]
            i_col_tmp = scaled_data.columns.get_loc(col_tmp)

            for k_v in results[0].items():
                col_dist_counts_global[k_v[0]] = k_v[1]
            cols_outliers_global.extend(results[1])

            scaled_data[col_tmp] = results[2]

            tmp_err_dist = results[4]
            new_error_dist.typos_values[col_tmp] = tmp_err_dist.typos_values.get(col_tmp, dict())
            new_error_dist.typos_num[col_tmp] = tmp_err_dist.typos_num.get(col_tmp, 0)
            new_error_dist.mv_values[col_tmp] = tmp_err_dist.mv_values.get(col_tmp, dict())
            new_error_dist.mv_num[col_tmp] = tmp_err_dist.mv_num.get(col_tmp, 0)
            new_error_dist.distinct_replacements_count[col_tmp] = tmp_err_dist.distinct_replacements_count.get(col_tmp, 0)
            new_error_dist.replacements_dicts[col_tmp] = tmp_err_dist.replacements_dicts.get(col_tmp, dict())
            new_error_dist.initial_difference_mask[:, i_col_tmp] = tmp_err_dist.initial_difference_mask[:, i_col_tmp]
            new_error_dist.mv_mask[:, i_col_tmp] = tmp_err_dist.mv_mask[:, i_col_tmp]
            new_error_dist.outliers_num[col_tmp] = tmp_err_dist.outliers_num.get(col_tmp, 0)
            new_error_dist.outliers_mask[col_tmp] = tmp_err_dist.outliers_mask.get(col_tmp, None)
    # pool.close()
    # pool.join()

    # Swaps numerical
    num_swaps = sum([sum(d.values()) for d in err_dist.swaps_dict_nums.values()]) * scaling_factor
    if num_swaps > 0 and len(err_dist.swaps_dict_nums) > 0:
        swap_dict_num_scaled, scaled_data, swaps_num_mask = AddSwaps().run(scaled_data=scaled_data, num_swaps=num_swaps,
                                                                           scaling_factor=scaling_factor,
                                                                           swaps_dict_dirty=err_dist.swaps_dict_nums)
        new_error_dist.swaps_col_count = swaps_num_mask.sum(axis=0) \
            if new_error_dist.swaps_col_count is None else new_error_dist.swaps_col_count + swaps_num_mask.sum(axis=0)

        new_error_dist.swaps_col_unique_count = (scaled_data[swaps_num_mask].nunique()).to_dict()
        # scaled_data[swaps_num_mask].nunique() * (new_error_dist.swaps_col_count != 0)).to_dict()
        new_error_dist.swaps_dict_nums = swap_dict_num_scaled

    # Swap str-str, str-numeric
    num_swaps = sum([sum(d.values()) for d in err_dist.swaps_dict_with_str.values()]) * scaling_factor
    if num_swaps > 0 and len(err_dist.swaps_dict_with_str) > 0:
        swap_dict_str_scaled, scaled_data, swaps_str_mask = AddSwaps().run(scaled_data=scaled_data, num_swaps=num_swaps,
                                                                           swaps_dict_dirty=err_dist.swaps_dict_with_str,
                                                                           scaling_factor=scaling_factor)

        new_error_dist.swaps_col_count = swaps_str_mask.sum(axis=0) \
            if new_error_dist.swaps_col_count is None else new_error_dist.swaps_col_count + swaps_str_mask.sum(axis=0)
        new_error_dist.swaps_col_unique_count = (scaled_data[swaps_str_mask].nunique()).to_dict()
                # scaled_data[swaps_num_mask].nunique() * (new_error_dist.swaps_col_count != 0)).to_dict()
        new_error_dist.swaps_dict_with_str = swap_dict_str_scaled

        # Compute column means
        new_error_dist.dirty_mean = utils.col_means(data=scaled_data, mask=~(swaps_str_mask | new_error_dist.initial_difference_mask),
                                                    selected_cols=cols_outliers_global, clean_mean=err_dist.clean_mean) \
            if len(cols_outliers_global) > 0 else new_error_dist.dirty_mean

    else:
        new_error_dist.dirty_mean = utils.col_means(data=scaled_data, mask=~new_error_dist.initial_difference_mask,
                                                    selected_cols=cols_outliers_global, clean_mean=err_dist.clean_mean) \
            if len(cols_outliers_global) > 0 else new_error_dist.dirty_mean

    tasks = list()

    for col in cols_outliers_global:
        kwargs = {"scaled_data": scaled_data, "scaled_schema": scaled_schema,
                  "new_col_mean": new_error_dist.dirty_mean[col],
                  "err_dist": err_dist, "col": col, "col_dist_counts_global": col_dist_counts_global,
                  "new_error_dist": new_error_dist, "scaling_factor": scaling_factor}

        tasks.append(
            pool.apply_async(_get_local_outliers, (), kwargs))  # , callback=_log_returns_local

        col_i += 1

    for t in tasks:
        results = t.get()

        if results:
            col_tmp = results[1]
            scaled_data[col_tmp] = results[0]
            new_error_dist.outliers_values[col_tmp] = results[2]

    pool.close()
    pool.join()

    return scaled_data, new_error_dist


def _get_local_outliers(scaled_data: pd.DataFrame, scaled_schema: typing.Dict, new_error_dist: ErrorDistribution,
                    err_dist: ErrorDistribution, scaling_factor: float, col: str, col_dist_counts_global: typing.Dict, new_col_mean):
    # Set outliers mask and values, recompute mean of new data
    num_outliers = round(err_dist.outliers_num[col] * scaling_factor)
    num_new_outliers = round(
        (len(err_dist.outliers_values[col]) * col_dist_counts_global[col][0]) / col_dist_counts_global[col][1])  # FIXME

    new_outliers_dict = AddOutliers().run(col_name=col, num_outliers=num_outliers,
                                          num_new_dist_outliers=num_new_outliers, err_dist=err_dist,
                                          mean_scaled=new_col_mean,
                                          new_num_rows=scaled_data.shape[0])

    # Set outliers
    i_outlier = 0
    for error_count in new_outliers_dict.items():
        scaled_data[col].loc[new_error_dist.outliers_mask[col][i_outlier: i_outlier + error_count[1]]] = error_count[0]
        i_outlier += error_count[1]
    return scaled_data[col], col, new_outliers_dict


def _get_local_errors_per_col(scaled_data_tmp_col: pd.DataFrame, scaled_schema: typing.Dict, new_error_dist: ErrorDistribution,
                    err_dist: ErrorDistribution, scaling_factor: float, i: int, col: str):
    col_dist_counts = dict()
    cols_outliers = list()

    # d1 = scaled_data[col].value_counts().to_dict()

    # No errors to introduce n the column
    if err_dist.num_errors_per_col[i] == 0:
        return

    # Distincts
    dist_dirty_num = err_dist.distinct_num[col]
    dist_clean_num = len(err_dist.distinct_values_clean[col])
    dist_estimate = err_dist.distincts_estimate[col]
    ratio = math.ceil(dist_clean_num / dist_dirty_num)

    # Check distints -> how many new values can be added +- ratio between clean/dirty
    ratio_add_distincts = round(err_dist.original_rows * scaling_factor) * ratio
    distincts_new_num = dist_estimate if dist_estimate > 0 else ratio_add_distincts

    col_dist_counts[col] = (distincts_new_num, dist_dirty_num)

    # Typos
    num_typos = round(err_dist.typos_num[col] * scaling_factor)  # overall num typos to introduce
    if num_typos > 0:
        # number of new distinct typos
        num_new_typos = err_dist.distinct_typos_estimate[col]
        if len(err_dist.typos_values[col]) > num_new_typos:
            num_new_typos = len(err_dist.typos_values[col])

        new_error_dist.typos_values[col] = \
            AddTypos().run(col_name=col, num_typos=num_typos,
                           num_new_dist_typos=num_new_typos, old_num_typos=err_dist.typos_num[col],
                           existing_typos=err_dist.typos_values[col],
                           distincts=list(err_dist.distinct_values_clean[col].keys()))

    new_error_dist.typos_num[col] = num_typos

    # Missing values
    num_mv = round(err_dist.mv_num[col] * scaling_factor)
    if num_mv > 0:
        # number of new distinct mv
        num_new_mv = err_dist.distinct_mv_estimate[col]

        if len(err_dist.mv_values[col]) > num_new_mv:
            num_new_mv = len(err_dist.mv_values[col])

        new_error_dist.mv_values[col], less_unique_values_then_needed = AddMissingValues().run(
            col_name=col, num_mv=num_mv, num_new_dist_mv=num_new_mv, old_num_mv=err_dist.mv_num[col],
            existing_mv=err_dist.mv_values[col])
        # errors_dict.update(new_mv_dict)
        # new_error_dist.mv_values[col] = new_mv_dict

    new_error_dist.mv_num[col] = num_mv

    # For random indices of errors
    indices_vector = list(range(0, scaled_data_tmp_col.shape[0]))

    # Replacements - don't change number of distincts
    replacements = round(err_dist.distinct_replacements_count[col] * scaling_factor)
    if replacements > 0:
        indices_values_replacements, new_column_replacement, replacements_dict, count_replacements = AddReplacements() \
            .run(scaled_column=scaled_data_tmp_col, col_name=col, num_replacements=replacements,
                 scaling_factor=scaling_factor, replacements_dict=err_dist.replacements_dicts[col])

        new_error_dist.distinct_replacements_count[col] = count_replacements
        scaled_data_tmp_col = new_column_replacement
        # new_error_dist.replacements_dicts.update(replacements_dict)
        new_error_dist.replacements_dicts[col] = replacements_dict

        # Remove already modified values
        indices_vector = list(filter(lambda index: index not in indices_values_replacements, indices_vector))

    num_outliers = round(err_dist.outliers_num[col] * scaling_factor)

    # Get indices for each error
    random.shuffle(indices_vector)
    indices_vector = indices_vector[0: (num_outliers + num_typos + num_mv + 1)]

    i_error = 0
    # Typos
    if num_typos > 0:
        for error_count in new_error_dist.typos_values[col].items():
            scaled_data_tmp_col.loc[indices_vector[i_error: i_error + error_count[1]]] = error_count[0]
            new_error_dist.initial_difference_mask[indices_vector[i_error: i_error + error_count[1]], i] = 1

            i_error += error_count[1]

    # Missing values
    if num_mv > 0:
        for error_count in new_error_dist.mv_values[col].items():
            scaled_data_tmp_col.loc[indices_vector[i_error: (i_error + error_count[1])]] = error_count[0]
            new_error_dist.mv_mask[indices_vector[i_error: (i_error + error_count[1])], i] = True
            new_error_dist.initial_difference_mask[indices_vector[i_error: (i_error + error_count[1])], i] = True

            i_error += error_count[1]

    # Outliers
    if num_outliers > 0 and len(err_dist.outliers_values[col]) > 0:
        new_error_dist.outliers_mask[col] = indices_vector[i_error:]
        new_error_dist.initial_difference_mask[indices_vector[i_error:], i] = 1
        new_error_dist.outliers_num[col] = num_outliers

        cols_outliers.append(col)

    # d2 = scaled_data[col].value_counts().to_dict()
    # set1 = set(d1.items())
    # set2 = set(d2.items())
    # print(set1 ^ set2)

    return col_dist_counts, cols_outliers, scaled_data_tmp_col, col, new_error_dist
