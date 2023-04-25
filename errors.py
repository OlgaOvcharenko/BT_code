import copy
import math
import random
import time
import typing
import uuid
from abc import abstractmethod
from typing import Tuple

import pyspark.pandas as ps
import pyspark.sql
import pyspark.sql.functions as F
from pandas import Series
from pyspark.sql.functions import lit, col

from scale_dataset import DataGenSP
from utils import *

pd.options.mode.chained_assignment = None


def get_and_save_mask(shape: Tuple, col, col_data, modified_col, file_path: str = 'masks/' + str(uuid.uuid4())):
    """Create mask and save into file"""

    mask = np.full(shape, False, dtype=bool)
    if isinstance(modified_col, Series):
        modified_col = modified_col.values

    if isinstance(col_data, Series):
        col_data = col_data.values

    mask[:, col] = modified_col.reshape((modified_col.shape[0],)) != col_data.reshape((col_data.shape[0],))

    # save binary mask into file in case needed
    save_mask(mask, file_path)

    return mask


def save_mask(mask, file_path: str = 'masks/' + str(uuid.uuid4())):
    """Save mask into file"""
    mask.dump(file_path)


def load_mask(file_path: str = 'masks/tmp_mask'):
    """Load mask from file"""
    return np.load(file_path)


class Transformations:
    def __init__(self):
        # self.new_data = new_data
        pass

    @abstractmethod
    def run(self, *args):
        raise NotImplementedError


class AddTypos(Transformations):
    def run(self, col_name, num_typos: int, num_new_dist_typos: int,
            old_num_typos: int, existing_typos: typing.Dict, distincts):
        """Add typos"""
        new_typos_dict = copy.deepcopy(existing_typos)

        num_old_dist_typos = len(existing_typos)
        while num_old_dist_typos != num_new_dist_typos:

            value = random.choice(list(distincts))

            # Randomly choose which typos to introduce
            typo_type = random.choices(range(0, 3), k=max(1, min(round(len(value) / 2), 3)))

            if isinstance(value, str):
                # Add 2 simple transformations or till dist diff condition
                i = 0
                while i < len(typo_type):  # FIXME  and len(value) / 2 > levenshtein(value, old_value)
                    # Swap 2 neighbouring chars
                    if typo_type[i] == 0:
                        if len(value) > 2:
                            swap_ix1 = np.random.randint(low=0, high=len(value) - 1)
                            if swap_ix1 == 0:
                                swap_ix2 = 1
                            elif swap_ix1 == len(value) - 1:
                                swap_ix2 = swap_ix1 - 1
                            else:
                                swap_ix2 = random.randint(swap_ix1 - 1, swap_ix1 + 1)

                            char1, char2 = value[swap_ix1:swap_ix1 + 1], value[swap_ix2:swap_ix2 + 1]
                            char2 = random.choice([chr(i) for i in range(128) if chr(i) != char1]) \
                                if char2 == char1 else char2
                            value = value[:swap_ix1] + char2 + value[swap_ix1 + 1:]
                            value = value[:swap_ix2] + char1 + value[swap_ix2 + 1:]
                        else:
                            char_to_add = random.choice([chr(i) for i in range(128)])
                            value = value + char_to_add

                    # Remove 1 char
                    elif typo_type[i] == 1:
                        if len(value) > 2:
                            remove_ix = np.random.randint(low=0, high=len(value) - 1)
                            value = value[:remove_ix] + value[remove_ix + 1:]

                        else:
                            char_to_add = random.choice([chr(i) for i in range(128)])
                            value = value + char_to_add

                    # Add 1 char on some position
                    elif typo_type[i] == 2:
                        if len(value) > 2:
                            rand_ix = np.random.randint(low=0, high=len(value) - 1)
                            char_to_add = random.choice([chr(i) for i in range(128)])
                            value = value[:rand_ix] + char_to_add + value[rand_ix:]
                        else:
                            char_to_add = random.choice([chr(i) for i in range(128)])
                            value = value + char_to_add

                    i += 1
            else:
                raise Exception('add_typos: Typos can be introduced only to str.')

            num_old_dist_typos = num_old_dist_typos + 1
            new_typos_dict[value] = 0

        min_frequency = min(list(existing_typos.values()))
        check_num_typos = 0

        for k in new_typos_dict.keys():
            ratio = (new_typos_dict[k] + min_frequency) / ((len(new_typos_dict) * min_frequency) + old_num_typos)
            new_typo_num = math.ceil(ratio * num_typos)

            new_typos_dict[k] = abs(new_typo_num)
            check_num_typos += new_typo_num

        if check_num_typos > num_typos:
            for k_v in new_typos_dict.items():
                if k_v[1] > 1:
                    new_typos_dict[k_v[0]] = k_v[1] - 1
                    check_num_typos -= 1
                if check_num_typos == num_typos:
                    return new_typos_dict

        if check_num_typos < num_typos:
            for k_v in new_typos_dict.items():
                new_typos_dict[k_v[0]] = k_v[1] + 1
                check_num_typos += 1
                if check_num_typos == num_typos:
                    return new_typos_dict

        return new_typos_dict

    def run_sp(self, col_name, num_typos: int, num_new_dist_typos: int,
            old_num_typos: int, existing_typos: typing.Dict, distincts):
        """Add typos"""
        new_typos_dict = copy.deepcopy(existing_typos)

        num_old_dist_typos = len(existing_typos)
        while num_old_dist_typos != num_new_dist_typos:

            value = random.choice(list(distincts))

            # Randomly choose which typos to introduce
            typo_type = random.choices(range(0, 3), k=max(1, min(round(len(value) / 2), 3)))

            if isinstance(value, str):
                # Add 2 simple transformations or till dist diff condition
                i = 0
                while i < len(typo_type):  # FIXME  and len(value) / 2 > levenshtein(value, old_value)
                    # Swap 2 neighbouring chars
                    if typo_type[i] == 0:
                        if len(value) > 2:
                            swap_ix1 = np.random.randint(low=0, high=len(value) - 1)
                            if swap_ix1 == 0:
                                swap_ix2 = 1
                            elif swap_ix1 == len(value) - 1:
                                swap_ix2 = swap_ix1 - 1
                            else:
                                swap_ix2 = random.randint(swap_ix1 - 1, swap_ix1 + 1)

                            char1, char2 = value[swap_ix1:swap_ix1 + 1], value[swap_ix2:swap_ix2 + 1]
                            char2 = random.choice([chr(i) for i in range(128) if chr(i) != char1]) \
                                if char2 == char1 else char2
                            value = value[:swap_ix1] + char2 + value[swap_ix1 + 1:]
                            value = value[:swap_ix2] + char1 + value[swap_ix2 + 1:]
                        else:
                            char_to_add = random.choice([chr(i) for i in range(128)])
                            value = value + char_to_add

                    # Remove 1 char
                    elif typo_type[i] == 1:
                        if len(value) > 2:
                            remove_ix = np.random.randint(low=0, high=len(value) - 1)
                            value = value[:remove_ix] + value[remove_ix + 1:]

                        else:
                            char_to_add = random.choice([chr(i) for i in range(128)])
                            value = value + char_to_add

                    # Add 1 char on some position
                    elif typo_type[i] == 2:
                        if len(value) > 2:
                            rand_ix = np.random.randint(low=0, high=len(value) - 1)
                            char_to_add = random.choice([chr(i) for i in range(128)])
                            value = value[:rand_ix] + char_to_add + value[rand_ix:]
                        else:
                            char_to_add = random.choice([chr(i) for i in range(128)])
                            value = value + char_to_add

                    i += 1
            else:
                raise Exception('add_typos: Typos can be introduced only to str.')
            num_old_dist_typos = num_old_dist_typos + 1
            new_typos_dict[value] = 0

        min_frequency = min(list(existing_typos.values()))
        check_num_typos = 0

        for k in new_typos_dict.keys():
            ratio = (new_typos_dict[k] + min_frequency) / ((len(new_typos_dict) * min_frequency) + old_num_typos)
            new_typo_num = round(ratio * num_typos)

            new_typos_dict[k] = abs(new_typo_num)
            check_num_typos += new_typo_num

        if check_num_typos > num_typos:
            for k_v in new_typos_dict.items():
                if k_v[1] > 1:
                    new_typos_dict[k_v[0]] = k_v[1] - 1
                    check_num_typos -= 1
                if check_num_typos == num_typos:
                    return new_typos_dict

        if check_num_typos < num_typos:
            for k_v in new_typos_dict.items():
                new_typos_dict[k_v[0]] = k_v[1] + 1
                check_num_typos += 1
                if check_num_typos == num_typos:
                    return new_typos_dict

        return new_typos_dict


class AddMissingValues(Transformations):
    def run(self, col_name, num_mv: int, num_new_dist_mv: int, old_num_mv: int, existing_mv: typing.Dict):
        """Add MVs"""
        less_unique = False
        new_mv_dict = copy.deepcopy(existing_mv)
        num_old_dist_mv = len(existing_mv)

        mv_to_take = num_new_dist_mv - num_old_dist_mv if num_new_dist_mv > num_old_dist_mv else num_old_dist_mv

        new_dist_mv = set(random.choices(list(set(error_distribution.MISSING_VALUES).difference(
            set(existing_mv.keys()))), k=abs(mv_to_take)))

        for k in new_dist_mv:
            new_mv_dict[k] = 0

        if len(new_dist_mv) + num_old_dist_mv < num_new_dist_mv:
            less_unique = True
            print(f"INFO: Added less new unique MV ({len(new_dist_mv) + num_old_dist_mv}) "
                  f"then expected unique MV ({num_new_dist_mv}).")

        min_frequency = min(list(existing_mv.values()))
        check_num_mv = 0

        for k in new_mv_dict.keys():
            ratio = (new_mv_dict[k] + min_frequency) / ((len(new_mv_dict) * min_frequency) + old_num_mv)
            new_mv_num = math.floor(ratio * num_mv)

            new_mv_dict[k] = new_mv_num
            check_num_mv += new_mv_num

            # # To check rounds after
            # if new_mv_num > max_num:
            #     max_count_typo_key = k
        # print(f"Num mv {check_num_mv}")

        if check_num_mv > num_mv:
            for k_v in new_mv_dict.items():
                if k_v[1] > 1:
                    new_mv_dict[k_v[0]] = k_v[1] - 1
                    check_num_mv -= 1
                if check_num_mv == num_mv:
                    return new_mv_dict, less_unique

        if check_num_mv < num_mv:
            for k_v in new_mv_dict.items():
                new_mv_dict[k_v[0]] = k_v[1] + 1
                check_num_mv += 1
                if check_num_mv == num_mv:
                    return new_mv_dict, less_unique

        return new_mv_dict, less_unique


class AddOutliers(Transformations):
    def run(self, col_name, num_outliers: int, num_new_dist_outliers: int,
            err_dist: error_distribution.ErrorDistribution, mean_scaled: float, new_num_rows: int):
        """Add outliers"""

        # Count number of new values to introduce
        num_old_dist_outliers = len(err_dist.outliers_values[col_name])

        outlier_values = dict()

        half_to_introduce = math.floor(num_new_dist_outliers / 2) if num_new_dist_outliers > 1 else 1
        outlier_types = random.choices(range(0, 7), k=half_to_introduce)

        # To shift means
        correction = ((err_dist.dirty_mean[col_name] - mean_scaled) * new_num_rows) / num_outliers

        q1, q3 = err_dist.clean_q25[col_name], err_dist.clean_q75[col_name]
        iqr = q3 - q1
        lower_limit, upper_limit = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        if num_new_dist_outliers == 1 and (upper_limit <= correction or lower_limit <= correction):
            outlier_values[correction] = num_outliers
            return outlier_values

        # Recompute limits for later balancing: max distance to mean, holds for both sides
        dist_from_mean_to_outliers = max(abs(upper_limit - err_dist.dirty_mean[col_name]), abs(err_dist.dirty_mean[col_name] - lower_limit))
        lower_limit, upper_limit = err_dist.clean_mean[col_name] - dist_from_mean_to_outliers, err_dist.clean_mean[col_name] + dist_from_mean_to_outliers

        # Reflect around zero and subtract correction
        distance_to_zero = max(abs(lower_limit), abs(upper_limit))
        lower_limit, upper_limit = (-1 * distance_to_zero) - correction, distance_to_zero - correction

        # Frequency per outlier
        num_per_outlier = math.ceil(num_outliers / num_new_dist_outliers)

        # Rounding error
        num_missed_outliers = 0
        if (num_per_outlier * num_new_dist_outliers) != num_outliers:
            num_missed_outliers = num_outliers - (num_per_outlier * num_new_dist_outliers)

        i = 0
        for outlier_type in outlier_types:

            # Get an outlier
            # print(err_dist.distinct_values_clean[col_name].keys())
            value = random.choice(list(err_dist.distinct_values_clean[col_name].keys()))
            new_outlier_value = AddOutliers.get_outlier(outlier_type=outlier_type, err_dist=err_dist,
                                                        lower_limit=lower_limit, upper_limit=upper_limit,
                                                        value=value, col_name=col_name, correction=correction)

            # Find balanced value on another side of mean to balance it
            balanced_val = -1 * new_outlier_value
            # balanced_val = err_dist.clean_mean[col_name] + abs(err_dist.clean_mean[col_name] - new_outlier_value) \
            #     if new_outlier_value < err_dist.clean_mean[col_name] else \
            #     err_dist.clean_mean[col_name] - abs(err_dist.clean_mean[col_name] - new_outlier_value)
            # balanced_val = err_dist.schema[col_name](balanced_val)

            # print(f"New value {new_outlier_value}")
            # print(f"Balanced value {balanced_val}")

            new_outlier_value = int(new_outlier_value + correction) if err_dist.schema[col_name] is np.int else new_outlier_value + correction
            balanced_val = int(balanced_val + correction) if err_dist.schema[col_name] is np.int else balanced_val + correction

            i += 1
            if i == (half_to_introduce - 1) and num_missed_outliers > 0:
                # Add aligned to two sides outliers
                if num_missed_outliers > 1:
                    to_add_number = math.floor(num_missed_outliers / 2)
                    outlier_values[balanced_val] = num_per_outlier + to_add_number
                    outlier_values[new_outlier_value] = num_per_outlier + to_add_number

                #  Balance 1 outlier
                if num_missed_outliers == 1 or num_missed_outliers % 2 != 0:
                    left = new_outlier_value if new_outlier_value < err_dist.clean_mean[col_name] else balanced_val
                    right = new_outlier_value if new_outlier_value > err_dist.clean_mean[col_name] else balanced_val

                    # New shifted value
                    left = err_dist.schema[col_name]((num_per_outlier * left - right) / num_per_outlier)

                    outlier_values[left] = num_per_outlier
                    outlier_values[right] = num_per_outlier + 1

            # Odd number of distincts (e.g., 3 distincts - 100 num_outliers)
            elif i == half_to_introduce and num_new_dist_outliers % 2 != 0:
                new_outlier_value = err_dist.schema[col_name]\
                    (err_dist.clean_mean[col_name] + 2 * abs(err_dist.clean_mean[col_name] - new_outlier_value)
                     if new_outlier_value > err_dist.clean_mean[col_name] else
                     err_dist.clean_mean[col_name] - 2 * abs(err_dist.clean_mean[col_name] - new_outlier_value))

                eps = 0.00000001 * balanced_val
                outlier_values[balanced_val - eps] = num_per_outlier
                outlier_values[balanced_val + eps] = num_per_outlier

                outlier_values[new_outlier_value] = num_per_outlier

            else:
                outlier_values[balanced_val] = num_per_outlier
                outlier_values[new_outlier_value] = num_per_outlier

        return outlier_values

    @staticmethod
    def get_outlier(outlier_type: int, value, err_dist: error_distribution.ErrorDistribution, col_name,
                                  correction, lower_limit, upper_limit):
        if outlier_type == 0:
            new_outlier_value = err_dist.clean_mean[col_name] + \
                                (abs(err_dist.clean_max[col_name] - err_dist.clean_min[col_name]) + abs(correction))
            if not(lower_limit < new_outlier_value < upper_limit):
                new_outlier_value = upper_limit + correction

        elif outlier_type == 1:
            new_outlier_value = err_dist.clean_mean[col_name] - \
                                (abs(err_dist.clean_max[col_name] - err_dist.clean_min[col_name]) + abs(correction))
            if not(lower_limit < new_outlier_value < upper_limit):
                new_outlier_value = lower_limit - correction

        elif outlier_type == 2:
            while lower_limit < value < upper_limit:
                value = value * 10
            new_outlier_value = value

        elif outlier_type == 3:
            new_outlier_value = lower_limit - abs(value)

        elif outlier_type == 4:
            new_outlier_value = upper_limit + abs(value)

        elif outlier_type == 5:
            upper_limit_1 = (err_dist.clean_mean[col_name] + 3 * math.sqrt(err_dist.clean_var[col_name]))
            new_outlier_value = upper_limit_1 + abs(err_dist.clean_min[col_name])

        elif outlier_type == 6:
            lower_limit = err_dist.clean_mean[col_name] - 3 * math.sqrt(err_dist.clean_var[col_name])
            new_outlier_value = lower_limit - abs(err_dist.clean_min[col_name])
        new_outlier_value = new_outlier_value
        return new_outlier_value


class AddReplacements(Transformations):
    def run_sp(self, scaled_column: pyspark.sql.DataFrame, col_name, num_replacements: int, scaling_factor: float,
               replacements_dict: typing.Dict, err_dist: error_distribution.ErrorDistribution, index_col: str):
        """Add replacements"""

        repl_tmp_path = "tmp/" + col_name + "_replacement_tmp"
        col_name_modified = col_name + "_modified"
        scaled_column = scaled_column\
            .repartition(DataGenSP.get_number_partitions(err_dist.generated_rows, 1), col_name)\
            .cache()

        print("\n")

        replacements_indices_list = list()

        if static_utils.is_in_cluster():
            hdfs_client = static_utils.write_to_hdfs()
            with hdfs_client.write(repl_tmp_path, encoding="utf-8") as file_errors:
                tmp_replacements = 0

                csv_buf = f"{index_col},{col_name_modified}\n"

                for key_val in replacements_dict.items():
                    k, v_dict = (key_val[0]).item() if not isinstance(key_val[0], str) else key_val[0], key_val[1]
                    sum_replacements = round(sum(v_dict.values()) * scaling_factor)

                    # limit_time = time.time()

                    # Filter and limit FIXME  .sample(fraction=fraction) \
                    fraction = sum_replacements / err_dist.distinct_values_clean[col_name][k]
                    filtered_values = scaled_column\
                        .filter(pyspark.sql.functions.col(col_name) == k)\
                        .limit(sum_replacements)\
                        .select(col(index_col))
                    filtered_indices = [r[0] for r in filtered_values.toLocalIterator()]

                    if replacements_indices_list:
                        replacements_indices_list.extend(filtered_indices)
                    else:
                        replacements_indices_list = filtered_indices

                    tmp_replacements += len(filtered_indices)

                    dict_items = list(v_dict.items())
                    dict_i, start, stop = 0, 0, dict_items[0][1]
                    for row_i, index_from_data in enumerate(filtered_indices):
                        csv_str = f"{index_from_data},{dict_items[dict_i][0]}\n"
                        csv_buf += csv_str

                        if len(csv_buf) > 2000000:
                            file_errors.write(csv_buf)
                            csv_buf = ""

                        if start == stop:
                            dict_i += 1
                            if dict_i > len(dict_items):
                                break
                            stop = dict_items[dict_i][1]

                if len(csv_buf) > 0:
                    file_errors.write(csv_buf)
                    csv_buf = ""

                    # print(f"REPLACEMENTS: Write to the file {time.time() - limit_time}")

        else:
            with open(repl_tmp_path, "w") as file_errors:
                csv_buf = f"{index_col},{col_name_modified}\n"

                tmp_replacements = 0
                for key_val in replacements_dict.items():
                    k, v_dict = (key_val[0]).item() if not isinstance(key_val[0], str) else key_val[0], key_val[1]
                    sum_replacements = math.floor(sum(v_dict.values()) * scaling_factor)

                    # limit_time = time.time()

                    # Filter and limit FIXME  .sample(fraction=fraction) \
                    # fraction = sum_replacements / err_dist.distinct_values_clean[col_name][k]  # FIXME increase
                    filtered_values = scaled_column \
                        .filter(pyspark.sql.functions.col(col_name) == k) \
                        .limit(sum_replacements) \
                        .select(col(index_col))
                    filtered_indices = [r[0] for r in filtered_values.toLocalIterator()]

                    if replacements_indices_list:
                        replacements_indices_list.extend(filtered_indices)
                    else:
                        replacements_indices_list = filtered_indices

                    # if len(filtered_indices) != sum_replacements:
                    #     print(col_name)
                    #     print(k)
                    #     (scaled_column.select(col_name).printSchema())
                    #     # scaled_column.groupBy(col_name).count().show(n=10000)
                    #     print(v_dict.values())
                    #     print(f"Filter only {scaled_column.filter(pyspark.sql.functions.col(col_name) == k).count()}")
                    #     print(f"Sum replacements {(sum_replacements)}\n\n")
                    # # print(f"Filter and limit took {time.time() - limit_time}")

                    tmp_replacements += len(filtered_indices)

                    dict_items = list(v_dict.items())
                    dict_i, start, stop = 0, 0, dict_items[0][1]
                    for row_i, index_from_data in enumerate(filtered_indices):
                        csv_str = f"{index_from_data},{dict_items[dict_i][0]}\n"
                        csv_buf += csv_str

                        if len(csv_buf) > 2000000:
                            file_errors.write(csv_buf)
                            csv_buf = ""

                        if start == stop:
                            dict_i += 1
                            if dict_i > len(dict_items):
                                break
                            stop = dict_items[dict_i][1]

                    # print(f"REPLACEMENTS: Write to the file {time.time() - limit_time}")
                if len(csv_buf) > 0:
                    file_errors.write(csv_buf)
                    csv_buf = ""

        return repl_tmp_path, tmp_replacements, col_name_modified, replacements_indices_list

    def run_sp_joins(self, scaled_column: pyspark.sql.DataFrame, col_name, num_replacements: int, scaling_factor: float,
               replacements_dict: typing.Dict, err_dist: error_distribution.ErrorDistribution, index_col: str):
        """Add replacements"""

        all_changes_df = None
        col_modified_name = col_name + "_modified"

        for key_val in replacements_dict.items():
            k, v_dict = (key_val[0]).item() if not isinstance(key_val[0], str) else key_val[0], key_val[1]
            sum_replacements = round(sum(v_dict.values()))
            num_replacements_per_val_new = math.floor(sum_replacements * scaling_factor)

            limit_time = time.time()

            # Filter and limit
            filtered_values = scaled_column\
                .filter(pyspark.sql.functions.col(col_name) == k)

            fraction = sum_replacements / err_dist.distinct_values_clean[col_name][k]
            filtered_values = filtered_values \
                .sample(fraction=fraction)\
                .cache()
                # .repartition(16, index_col) \

            print(f"\nFilter and limit took {time.time() - limit_time}")

            count_unmodified = num_replacements_per_val_new
            for replacement_count in v_dict.items():
                start_time = time.time()
                num_new_replacements = round(replacement_count[1] * scaling_factor)

                #  Filter and set
                updated_filtered_values = filtered_values\
                    .filter(pyspark.sql.functions.col(col_name) == k)
                updated_filtered_values = updated_filtered_values \
                    .sample(fraction=num_new_replacements / count_unmodified) \
                    .select(pyspark.sql.functions.col(index_col), lit(replacement_count[0]).alias(col_modified_name))
                print(f"Filter subset, limit, add column {time.time() - start_time}")

                # Update filtered values and join
                new_cols = [index_col]
                append_val = F.when(updated_filtered_values[col_modified_name].isNull(), filtered_values[col_name])\
                    .otherwise(updated_filtered_values[col_modified_name]).alias(col_name)
                new_cols.append(append_val)
                print(f"Create new columns {time.time() - start_time}")

                filtered_values = filtered_values \
                    .join(updated_filtered_values, index_col, "left_outer") \
                    .select(new_cols)

                count_unmodified -= num_new_replacements

                print(f"Replacement done {time.time() - start_time}\n")

            if all_changes_df:
                all_changes_df.union(filtered_values)
            else:
                all_changes_df = filtered_values

        return all_changes_df, num_replacements, col_modified_name

    def run_sp_pandas(self, scaled_column: pyspark.pandas.DataFrame, col_name, num_replacements: int, scaling_factor: float,
               replacements_dict: typing.Dict, err_dist: error_distribution.ErrorDistribution):
        """Add replacements"""
        ps.set_option('compute.ops_on_diff_frames', True)

        # Clean and dirty dataset histograms
        # How to preserve the stats, distincts are already preserved
        # new_scaled_column = scaled_column.copy(deep=True)

        df_changes = ps.DataFrame(columns=[col_name])
        for key_val in replacements_dict.items():
            k, v_dict = (key_val[0]).item() if not isinstance(key_val[0], str) else key_val[0], key_val[1]
            sum_replacements = round(sum(v_dict.values()))
            num_replacements_per_val_new = sum_replacements * scaling_factor
            filtered_values = scaled_column.loc[scaled_column[col_name] == k]
            fraction = sum_replacements / err_dist.distinct_values_clean[col_name][k]
            filtered_values = filtered_values.sample(frac=fraction)

            count_unmodified = num_replacements_per_val_new
            for replacement_count in v_dict.items():
                # print(f"Key {k} \t val {replacement_count}")

                num_new_replacements = round(replacement_count[1] * scaling_factor)
                updated_filtered_values = filtered_values.loc[filtered_values[col_name] == k]
                updated_filtered_values = updated_filtered_values.sample(frac=num_new_replacements / count_unmodified)
                updated_filtered_values[col_name] = replacement_count[0]
                count_unmodified -= num_new_replacements

                filtered_values.update(updated_filtered_values)

            df_changes = df_changes.append(filtered_values)

        scaled_column.update(df_changes)

        tmp_replacements = df_changes.shape[0]
        if tmp_replacements != num_replacements:
            print(f"INFO: Rounding error between actual replacements added ({tmp_replacements}) "
                  f"and expected ({num_replacements}).")

        return scaled_column, tmp_replacements

    def run(self, scaled_column: pd.DataFrame, col_name, num_replacements: int, scaling_factor: float, replacements_dict: typing.Dict):
        """Add replacements"""
        # Clean and dirty dataset histograms
        # How to preserve the stats, distincts are already preserved
        tmp_replacements = 0
        new_replacements_dict = copy.deepcopy(replacements_dict)
        new_scaled_column = copy.deepcopy(scaled_column)

        indices_values = list()
        for key_val in new_replacements_dict.items():
            k, v_dict = key_val[0], key_val[1]
            num_replacements_per_val_new = math.floor(sum(v_dict.values()) * scaling_factor)

            # print(len(scaled_column.index[scaled_column == k].tolist()))
            # print((scaled_column.index[scaled_column == k].tolist()))
            indices_list = scaled_column.index[scaled_column == k].tolist()
            num_replacements_per_val_new = len(indices_list) if len(indices_list) < num_replacements_per_val_new \
                else num_replacements_per_val_new
            indices = random.choices(indices_list, k=num_replacements_per_val_new)

            tmp_replacements += num_replacements_per_val_new

            i = 0
            for replacement_count in v_dict.items():
                num_new_replacements = round(replacement_count[1] * scaling_factor)
                indices_per_replacement = indices[i:i + num_new_replacements]
                indices_values.extend(indices_per_replacement)
                new_scaled_column.loc[indices_per_replacement] = replacement_count[0]
                v_dict[replacement_count[0]] = num_new_replacements

                i += num_new_replacements

            new_replacements_dict[k] = v_dict

        if tmp_replacements != num_replacements:
            print(f"INFO: Rounding error between actual replacements added ({tmp_replacements}) "
                  f"and expected ({num_replacements}).")

        # mask = scaled_column != new_scaled_column

        return indices_values, new_scaled_column, new_replacements_dict, tmp_replacements


class AddSwaps(Transformations):
    def run_sp(
            self, scaled_data: pyspark.sql.DataFrame, num_swaps, scaling_factor: float,
            swaps_dict: typing.Dict, index_col: str, original_rows: int):
        """Add swaps"""
        print("\n")

        swaps_paths = list()
        col_names_modified = list()
        tmp_swaps = list()

        for columns_dict in swaps_dict.items():
            col_name1, col_name2, v_dict = columns_dict[0][0], columns_dict[0][1], columns_dict[1]

            swaps_tmp_path = "tmp/" + col_name1 + "_" + col_name2 + "_swaps_tmp"
            col_name1_modified, col_name2_modified = col_name1 + "_modified", col_name2 + "_modified"

            col_names_modified.append((col_name1_modified, col_name2_modified))

            # scaled_data = scaled_data.repartition(16, [col_name1, col_name2]).cache()

            swaps_paths.append(swaps_tmp_path)

            if static_utils.is_in_cluster():
                hdfs_client = static_utils.write_to_hdfs()
                with hdfs_client.write(swaps_tmp_path, encoding="utf-8") as file_errors:
                    csv_buf = f"{index_col},{col_name1_modified},{col_name2_modified}\n"
                    tmp_swap = 0

                    for swap_count in v_dict.items():
                        swap_clean, swap_dirty = (swap_count[0][0]).item() if not isinstance(swap_count[0][0], str) \
                                                     else swap_count[0][0], swap_count[0][1]
                        num_new_swaps = round(swap_count[1] * scaling_factor)

                        # limit_time = time.time()

                        # Filter and limit
                        filtered_values = scaled_data \
                            .select(col(index_col), col(col_name1), col(col_name2))
                        filtered_values = filtered_values\
                            .filter(filtered_values[col_name1] == swap_clean) \
                            .filter(filtered_values[col_name2] == swap_dirty) \
                            .select(col(index_col)) \
                            .limit(num_new_swaps)

                        filtered_indices = [r[0] for r in filtered_values.toLocalIterator()]
                        # print(f"Filter and limit took {time.time() - limit_time}, fraction length {len(filtered_indices)}")

                        tmp_swap += len(filtered_indices)

                        for row_i, index_from_data in enumerate(filtered_indices):
                            csv_str = f"{index_from_data},{swap_dirty},{swap_clean}\n"
                            csv_buf += csv_str

                            if len(csv_buf) > 2000000:
                                file_errors.write(csv_buf)
                                csv_buf = ""

                        # print(f"SWAPS: Write to the file {time.time() - limit_time}")

                    if len(csv_buf) > 0:
                        file_errors.write(csv_buf)
                        csv_buf = ""

                    tmp_swaps.append(tmp_swap)

            else:
                with open(swaps_tmp_path, "w") as file_errors:
                    csv_buf = f"{index_col},{col_name1_modified},{col_name2_modified}\n"
                    tmp_swap = 0

                    for swap_count in v_dict.items():
                        swap_clean, swap_dirty = (swap_count[0][0]).item() if not isinstance(swap_count[0][0], str) \
                                                     else swap_count[0][0], swap_count[0][1]
                        num_new_swaps = round(swap_count[1] * scaling_factor)

                        # limit_time = time.time()

                        # Filter and limit
                        filtered_values = scaled_data \
                            .select(col(index_col), col(col_name1), col(col_name2))
                        filtered_values = filtered_values \
                            .filter(filtered_values[col_name1] == swap_clean) \
                            .filter(filtered_values[col_name2] == swap_dirty) \
                            .select(col(index_col)) \
                            .limit(num_new_swaps)

                        filtered_indices = [r[0] for r in filtered_values.toLocalIterator()]
                        # print(
                        #     f"Filter and limit took {time.time() - limit_time}, fraction length {len(filtered_indices)}")

                        tmp_swap += len(filtered_indices)

                        for row_i, index_from_data in enumerate(filtered_indices):
                            csv_str = f"{index_from_data},{swap_dirty},{swap_clean}\n"
                            csv_buf += csv_str

                            if len(csv_buf) > 2000000:
                                file_errors.write(csv_buf)
                                csv_buf = ""

                        # print(f"SWAPS: Write to the file {time.time() - limit_time}")

                    if len(csv_buf) > 0:
                        file_errors.write(csv_buf)
                        csv_buf = ""

                    tmp_swaps.append(tmp_swap)

        return swaps_paths, tmp_swaps, col_names_modified

    def run(self, scaled_data: pd.DataFrame, num_swaps, scaling_factor: float, swaps_dict_dirty: typing.Dict):
        """Add swaps"""
        swaps_dict = copy.deepcopy(swaps_dict_dirty)

        tmp_swaps = 0

        mask = np.full(scaled_data.shape, False, dtype=bool)
        for columns_dict in swaps_dict.items():
            col_name1, col_name2, v_dict = columns_dict[0][0], columns_dict[0][1], columns_dict[1]

            for swap_count in v_dict.items():
                swap_clean, swap_dirty = swap_count[0][0], swap_count[0][1]
                num_new_swaps = round(swap_count[1] * scaling_factor)

                tmp_swaps += num_new_swaps

                indices = random.sample(list(scaled_data[col_name1].index[scaled_data[col_name1] == swap_clean]),
                                         k=num_new_swaps)
                swaps_dict[columns_dict[0]][swap_count[0]] = num_new_swaps

                #  Modify data
                scaled_data[col_name1].loc[indices] = swap_dirty
                scaled_data[col_name2].loc[indices] = swap_clean

                mask[indices, scaled_data.columns.get_loc(col_name1)] = True
                mask[indices, scaled_data.columns.get_loc(col_name2)] = True

        if tmp_swaps != num_swaps:
            print(f"INFO: Rounding error between actual replacements added ({tmp_swaps}) "
                  f"and expected ({num_swaps}).")

        return swaps_dict, scaled_data, mask

    def run_sp_pandas(self, scaled_data: pyspark.pandas.DataFrame, num_swaps, scaling_factor: float, swaps_dict_dirty: typing.Dict):
        """Add swaps"""
        swaps_dict = copy.deepcopy(swaps_dict_dirty)

        tmp_swaps = 0

        mask = np.full(scaled_data.shape, False, dtype=bool)
        for columns_dict in swaps_dict.items():
            col_name1, col_name2, v_dict = columns_dict[0][0], columns_dict[0][1], columns_dict[1]

            for swap_count in v_dict.items():
                swap_clean, swap_dirty = (swap_count[0][0]).item() if not isinstance(swap_count[0][0], str) \
                                             else swap_count[0][0], swap_count[0][1]
                num_new_swaps = round(swap_count[1] * scaling_factor)

                tmp_swaps += num_new_swaps

                indices = random.sample(scaled_data[col_name1].loc[scaled_data[col_name1] == swap_clean].index.tolist(),
                                        k=num_new_swaps)

                swaps_dict[columns_dict[0]][swap_count[0]] = num_new_swaps

                #  Modify data
                scaled_data[col_name1].loc[indices] = swap_dirty
                scaled_data[col_name2].loc[indices] = swap_clean

                mask[indices, scaled_data.columns.get_loc(col_name1)] = True
                mask[indices, scaled_data.columns.get_loc(col_name2)] = True

        if tmp_swaps != num_swaps:
            print(f"INFO: Rounding error between actual replacements added ({tmp_swaps}) "
                  f"and expected ({num_swaps}).")

        return swaps_dict, scaled_data, mask
