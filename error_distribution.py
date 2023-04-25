import copy
import math
import sys
import typing
from enum import Enum

import numpy as np
import pandas as pd
import pyspark.pandas
from pyspark.sql.functions import countDistinct, approx_count_distinct, col
from scipy.stats.mstats import mquantiles

import utils
from utils import Utils

MISSING_VALUES = [-9999, np.NAN, np.NaN, np.nan, np.inf, sys.float_info.max, sys.float_info.min,
                  "NA", "NaN", "NAN", "nan", "0-*", 99999]

# MISSING_VALUES = [-9999, np.NAN, np.NaN, np.nan, np.inf, sys.float_info.max, sys.float_info.min, 99999]


class OriginalErrorType(Enum):
    MV = 0
    OUTLIER = 1
    TYPO = 2
    DISTINCT = 3
    SWAP = 4
    REPLACEMENT = 5
    FD = 6


class ErrorDistribution:
    def __init__(self, schema: typing.Dict, rows: int, cols: int):
        self.original_rows = rows
        self.original_cols = cols
        self.generated_rows = 0  # size of the clean part on which to generate errors
        self.scaling_factor = 0

        self.schema = schema

        self.initial_difference_mask = np.full((rows, cols), False, dtype=bool)
        self.num_errors_per_col = dict()

        # Distincts
        self.distinct_values = dict()
        self.distinct_value_counts = dict()
        self.distinct_num = dict()

        self.distinct_values_clean = dict()

        # Estimated distincts in generated
        self.distincts_estimate = dict()

        # Missing values
        self.mv_mask = None
        self.mv_num = dict()
        self.mv_values = dict()
        self.distinct_mv_estimate = dict()

        # Outliers
        self.outliers_mask = dict()
        self.outliers_num = dict()
        self.outliers_values = dict()

        # Swaps
        self.swaps = dict()
        self.swaps_col_unique_count = dict()
        self.swaps_col_count = None
        # self.swaps_mask = None
        self.swaps_dict_nums = dict()
        self.swaps_dict_with_str = dict()

        # Typos
        self.typos_mask = None
        self.typos_num = dict()
        self.typos_values = dict()
        self.distinct_typos_estimate = dict()

        # Valid replacements from distinct set
        self.distinct_replacements_count = dict()
        self.replacements_dicts = dict()

        self.iqr_lower = dict()
        self.iqr_upper = dict()

        # Statistics clean
        self.clean_min = None
        self.clean_max = None
        self.clean_mean = None
        self.clean_mode = None

        self.clean_q25 = None
        self.clean_q50 = None
        self.clean_q75 = None

        self.clean_var = None
        self.clean_skew = None
        self.clean_kurtosis = None

        # Statistics dirty
        self.dirty_min = None
        self.dirty_max = None
        self.dirty_mean = None
        self.dirty_mode = None

        self.dirty_q25 = None
        self.dirty_q50 = None
        self.dirty_q75 = None

        self.dirty_var = None
        self.dirty_skew = None
        self.dirty_kurtosis = None

        self.FDs = None

    def get_scaled_stats_with_errors(self, scaling_factor: float):
        self.scaling_factor = scaling_factor

        self.mv_mask = np.full((self.original_rows, self.original_cols), False, dtype=bool)
        self.initial_difference_mask = np.full((self.original_rows, self.original_cols), False, dtype=bool)

        self.dirty_min = dict()
        self.dirty_max = dict()
        self.dirty_mean = dict()
        self.dirty_mode = dict()

        self.dirty_q25 = dict()
        self.dirty_q50 = dict()
        self.dirty_q75 = dict()

        self.dirty_var = dict()
        self.dirty_skew = dict()
        self.dirty_kurtosis = dict()
        self.swaps_col_count = np.zeros((1, self.original_cols))
        self.swaps_col_unique_count =dict()

    def get_error_dist_after_scaling_sp(self, generated: pyspark.sql.DataFrame, dirty_schema):
        self.num_errors_per_col = sum(self.initial_difference_mask)

        # Count distinct values
        # self.distinct_num = generated.agg(*(approx_count_distinct(col(c), rsd=0.1).alias(c)
        #                                        for c in generated.columns)).collect()[0].asDict()

        self.distinct_num = generated.agg(*(countDistinct(col(c)).alias(c)
                                            for c in generated.columns)).collect()[0].asDict()

        i = 0
        for c in generated.columns:
            # _________________________
            self.num_errors_per_col[i] = self.outliers_num.get(c, 0) + \
                                         self.mv_num.get(c, 0) + \
                                         self.distinct_replacements_count.get(c, 0) + \
                                         self.typos_num.get(c, 0)

            # _________________________
            # Typos
            if not (self.schema[c] in [np.str, np.object]):
                self.typos_num[c] = 0

            # For missing values that turn column into string
            # turn into NaNs if missing values, now float column
            if (self.schema[c] not in [np.str, np.object] and dirty_schema.dtype in [np.object, np.str])       \
                    or self.mv_num.get(c, 0) > 0:
                # col_numeric = ps.to_numeric(generated[c])
                # col_numeric = col_numeric.where(pd.Series(self.mv_mask[:, i]), 0).fillna(0)
                # self.dirty_mean[c] = col_numeric.mean()

                data_df = generated \
                    .select(c) \
                    .withColumn(c, generated[c].cast('float')) \
                    .na.fill(0)
                    # .replace(list(self.mv_values.keys()), 0)

                self.dirty_mean[c] = list(data_df
                                          .agg({c: "mean"})
                                          .collect()[0].asDict().values())[0]

                self.dirty_var[c] = list(data_df
                                          .agg({c: "variance"})
                                          .collect()[0].asDict().values())[0]

                # print(data_df.approxQuantile(c, [0.25, 0.5, 0.25], 0.25))
                # print(c)
                # tmp_q = data_df.approxQuantile(c, [0.25, 0.5, 0.25], 0.25)
                # if tmp_q:
                #     self.dirty_q25[c], self.dirty_q50[c], self.dirty_q75[c] = tmp_q
                # else:
                #     self.dirty_q25[c], self.dirty_q50[c], self.dirty_q75[c] = 0, 0, 0,

                self.dirty_min[c] = list(data_df
                                          .agg({c: "min"})
                                          .collect()[0].asDict().values())[0]

                self.dirty_max[c] = list(data_df
                                          .agg({c: "max"})
                                          .collect()[0].asDict().values())[0]

                # if np.isnan(self.dirty_mean[c]):
                #     self.dirty_mean[c] = 0
                #
                # if np.isnan(self.dirty_var[c]):
                #     self.dirty_var[c] = 0

            elif dirty_schema.dtype not in [np.object, np.str]:
                self.dirty_mean[c] = list(generated
                                          .agg({c: "mean"})
                                          .collect()[0].asDict().values())[0]

                self.dirty_var[c] = list(generated
                                         .agg({c: "variance"})
                                         .collect()[0].asDict().values())[0]

                # print(data_df.approxQuantile(c, [0.25, 0.5, 0.25], 0.25))
                # self.dirty_q25[c], self.dirty_q50[c], self.dirty_q75[c] = generated.approxQuantile(c, [0.25, 0.5, 0.25], 0.25)

                self.dirty_min[c] = list(generated
                                      .agg({c: "min"})
                                      .collect()[0].asDict().values())[0]

                self.dirty_max[c] = list(generated
                                      .agg({c: "max"})
                                      .collect()[0].asDict().values())[0]

                # if np.isnan(self.dirty_mean[c]):
                #     self.dirty_mean[c] = 0
                #
                # if np.isnan(self.dirty_var[c]):
                #     self.dirty_var[c] = 0
            i += 1


    def get_error_dist_after_scaling(self, generated: pd.DataFrame):
        self.num_errors_per_col = sum(self.initial_difference_mask)

        i = 0
        for c in generated.columns:
            # _________________________
            # Values distribution
            self.distinct_value_counts[c] = generated[c].value_counts().to_dict()

            # Distinct values counts
            self.distinct_values[c] = list(self.distinct_value_counts[c].keys())
            self.distinct_num[c] = len(self.distinct_values[c])

            # Outliers
            self.outliers_values[c] = [] if self.outliers_num.get(c, 0) == 0 else list(
                generated[c].loc[self.outliers_mask[c]].unique())

            # _________________________
            self.num_errors_per_col[i] = self.outliers_num.get(c, 0) + \
                                         self.mv_num.get(c, 0) + \
                                         self.distinct_replacements_count.get(c, 0) + \
                                         self.typos_num.get(c, 0)

            # _________________________
            # Typos
            if not (self.schema[c] in [np.str, np.object]):
                self.typos_num[c] = 0

            # For missing values that turn column into string
            # turn into NaNs if missing values, now float column

            if (self.schema[c] not in [np.str, np.object] and generated[c].dtype in [np.object, np.str]) \
                    or self.mv_num.get(c, 0) > 0:
                col_numeric = pd.to_numeric(generated[c].replace(to_replace=MISSING_VALUES, value=0), errors='coerce').fillna(0)

                quantiles = mquantiles(col_numeric)
                self.dirty_q25[c] = quantiles[0]
                self.dirty_q50[c] = quantiles[1]
                self.dirty_q75[c] = quantiles[1]

                self.dirty_skew[c] = col_numeric.skew()
                self.dirty_kurtosis[c] = col_numeric.kurtosis()

                tmp_agg = col_numeric.aggregate([np.min, np.max, np.mean, np.var])
                self.dirty_min[c] = tmp_agg.loc["amin"]
                self.dirty_max[c] = tmp_agg.loc["amax"]
                self.dirty_mean[c] = tmp_agg.loc["mean"]
                self.dirty_var[c] = tmp_agg.loc["var"]

            elif generated[c].dtype not in [np.object, np.str]:

                quantiles = mquantiles(generated[c])
                self.dirty_q25[c] = quantiles[0]
                self.dirty_q50[c] = quantiles[1]
                self.dirty_q75[c] = quantiles[1]

                self.dirty_skew[c] = generated[c].skew()
                self.dirty_kurtosis[c] = generated[c].kurtosis()

                tmp_agg = generated[c].aggregate([np.min, np.max, np.mean, np.var]).fillna(0)
                self.dirty_min[c] = tmp_agg.loc["amin"]
                self.dirty_max[c] = tmp_agg.loc["amax"]
                self.dirty_mean[c] = tmp_agg.loc["mean"]
                self.dirty_var[c] = tmp_agg.loc["var"]

            i += 1

        self.num_errors_per_col = self.num_errors_per_col + self.swaps_col_count

        print(f"Error count {self.num_errors_per_col}")

    def get_error_dist(self, clean_data: pd.DataFrame, dirty_data: pd.DataFrame, scaling_factor: float, fds_path: str = None):
        self.scaling_factor = scaling_factor
        self.generated_rows = round(self.original_rows * scaling_factor)

        # self.FDs = Utils.discoverFD(clean_data, 0.8) if not fds_path else utils.parse_fds_from_file(fds_path, clean_data.shape[1])   # FIXME
        self.FDs = None if not fds_path else utils.parse_fds_from_file(fds_path, clean_data.shape[1])

        # _________________________
        # Missing values mask
        self.mv_mask = np.full(dirty_data.shape, False, dtype=bool)
        for value in MISSING_VALUES:
            self.mv_mask = self.mv_mask | np.ma.getmask(np.ma.masked_values(dirty_data, value))

        # if value is not faulty, then it can't be MV -> not always true & self.initial_difference_mask
        self.mv_mask = self.mv_mask | dirty_data.isnull()  # np.array()
        self.mv_num = self.mv_mask.sum().to_dict()

        touched_cells = self.mv_mask

        self.initial_difference_mask = np.full(dirty_data.shape, False, dtype=bool)

        # Dict for swaps later
        dict_mv_for_swaps = dict()

        i = 0
        for c in dirty_data.columns:

            # For missing values that turn column into string
            # turn into NaNs if missing values, now float column
            if self.schema[c] not in [np.str, np.object] and dirty_data[c].dtype in [np.object, np.str]:
                dict_mv_for_swaps[c] = copy.deepcopy(dirty_data[c])
                pd.to_numeric(dirty_data[c], errors='coerce')
                dirty_data[c] = pd.to_numeric(dirty_data[c], errors='coerce')
                self.mv_mask[c] = self.mv_mask[c] | dirty_data[c].isnull().values

            self.initial_difference_mask[:, i] = np.array(clean_data[c] != dirty_data[c])

            # _________________________
            # Values distribution
            self.distinct_value_counts[c] = dirty_data[c].value_counts(dropna=False).to_dict()

            # Distinct values counts
            self.distinct_values[c] = list(self.distinct_value_counts[c].keys())
            self.distinct_values_clean[c] = clean_data[c].value_counts().to_dict()
            self.distinct_num[c] = len(self.distinct_values[c])

            # _________________________
            # Estimate distincts number for each col
            self.distincts_estimate[c] = max(self._distinct_count_schlosser(
                num_vals=self.distinct_num[c], freq_counts=list(self.distinct_value_counts[c].values()),
                n_rows=self.generated_rows, sample_size=self.original_rows),
                self._distinct_count_haas_strokes(num_vals=self.distinct_num[c], freq_counts=list(self.distinct_value_counts[c].values()),
                                                  n_rows=self.generated_rows, sample_size=self.original_rows))

            # _________________________
            # Outliers
            if self.schema[c] not in [np.str, np.object]:
                outliers_mask_iqr, lower_limit, upper_limit = \
                    Utils.detect_col_outliers_by_IQR(col_dirty=dirty_data[c], col_clean=clean_data[c], replace_outliers=False)
                outliers_mask_clean = (clean_data[c].max() < dirty_data[c]) + (clean_data[c].min() > dirty_data[c])
                outliers_mask_dist, lower_limit2, upper_limit2 = Utils.detect_col_outliers_from_dist(
                    col_dirty=dirty_data[c], col_clean=clean_data[c], replace_outliers=False)

                self.iqr_lower[c] = lower_limit
                self.iqr_upper[c] = upper_limit

                self.outliers_mask[c] = (outliers_mask_iqr | outliers_mask_clean | outliers_mask_dist) & \
                                        (~touched_cells[c]) & self.initial_difference_mask[:, i]

                # touched_cells[c] |= self.outliers_mask[c]

            else:
                self.outliers_values[c] = {}
                self.outliers_num[c] = 0
                self.outliers_mask[c] = np.full(dirty_data[c].shape, False, dtype=bool)

            i += 1

        self.num_errors_per_col = sum(self.initial_difference_mask)
        self.mv_num = self.mv_mask.sum().to_dict()

        # _________________________
        # Swaps
        swaps_mask = np.full(dirty_data.shape, False, dtype=bool)
        possible_swaps_rows = self.initial_difference_mask & (~touched_cells)
        possible_swaps_row_indices = [i for i, x in enumerate((possible_swaps_rows.sum(axis=1) > 1)) if x]

        for r in possible_swaps_row_indices:
            row_mask = possible_swaps_rows.loc[r]

            dirty_v = dirty_data.loc[r]
            clean_v = clean_data.loc[r]

            col_indices = [j for j, y in enumerate((dirty_v != clean_v) & row_mask) if y]

            for ix in [(a, b) for idx, a in enumerate(col_indices) for b in col_indices[idx + 1:]]:
                col1, col2 = clean_data.columns[ix[0]], clean_data.columns[ix[1]]
                tuple_key = (col1, col2)
                values_key = (clean_v[ix[0]], dirty_v[ix[0]])

                # Get values
                dirty_v1 = utils.float_or_str(dict_mv_for_swaps[col1].loc[r]
                                              if dict_mv_for_swaps.get(col1) is not None else dirty_v[ix[0]])
                dirty_v2 = utils.float_or_str(dict_mv_for_swaps[col2].loc[r]
                                              if dict_mv_for_swaps.get(col2) is not None else dirty_v[ix[1]])
                clean_v1 = utils.float_or_str(clean_v[ix[0]])
                clean_v2 = utils.float_or_str(clean_v[ix[1]])

                if dirty_v1 == clean_v2 and clean_v1 == dirty_v2:
                    swaps_mask[r, ix[0]] = True
                    swaps_mask[r, ix[1]] = True
                    self.outliers_mask[col1][r] = False
                    self.outliers_mask[col2][r] = False

                    if self.schema[col1] not in [np.str, np.object] and \
                            self.schema[col2] not in [np.str, np.object]:
                        if self.swaps_dict_nums.get(tuple_key):
                            if (self.swaps_dict_nums.get(tuple_key)).get(values_key):
                                (self.swaps_dict_nums.get(tuple_key))[values_key] += 1
                            else:
                                (self.swaps_dict_nums.get(tuple_key))[values_key] = 1

                        else:
                            self.swaps_dict_nums[tuple_key] = {values_key: 1}

                    else:
                        if self.swaps_dict_with_str.get(tuple_key):
                            if (self.swaps_dict_with_str.get(tuple_key)).get(values_key):
                                (self.swaps_dict_with_str.get(tuple_key))[values_key] += 1
                            else:
                                (self.swaps_dict_with_str.get(tuple_key))[values_key] = 1

                        else:
                            self.swaps_dict_with_str[tuple_key] = {values_key: 1}
        dict_mv_for_swaps.clear()

        self.swaps_col_count = swaps_mask.sum(axis=0)
        self.swaps_col_unique_count = (dirty_data[swaps_mask].nunique() * (self.swaps_col_count != 0)).to_dict()

        touched_cells |= swaps_mask

        # _________________________
        # Typos
        self.typos_mask = np.full(dirty_data.shape, False, dtype=bool)

        # _________________________
        # Outliers
        # Outliers mask

        i = 0
        for c in dirty_data.columns:
            # Missing values within the column
            self.mv_values[c] = {} if self.mv_num[c] == 0 \
                else dirty_data[c].loc[self.mv_mask[c]].value_counts(dropna=False).to_dict()

            # Outliers
            self.outliers_num[c] = self.outliers_mask[c].sum()
            self.outliers_values[c] = [] if self.outliers_num[c] == 0 else list(
                dirty_data[c].loc[self.outliers_mask[c]].unique())

            touched_cells[c] |= self.outliers_mask[c]

            # _________________________
            # Replacements within the column distincts, like Y-> N
            replacement = self.initial_difference_mask[:, i] & (~touched_cells[c]) & dirty_data[c].isin(list(self.distinct_values_clean[c].keys()))
            self.distinct_replacements_count[c] = sum(replacement)

            replacements_row_indices = [i for i, x in enumerate(replacement) if x]
            replacements_dict_values_count = dict()
            for r in replacements_row_indices:
                dirty_v = dirty_data[c].loc[r]
                clean_v = clean_data[c].loc[r]

                if replacements_dict_values_count.get(clean_v):
                    if (replacements_dict_values_count.get(clean_v)).get(dirty_v):
                        (replacements_dict_values_count[clean_v])[dirty_v] += 1
                    else:
                        (replacements_dict_values_count[clean_v])[dirty_v] = 1

                else:
                    replacements_dict_values_count[clean_v] = {dirty_v: 1}

            self.replacements_dicts[c] = replacements_dict_values_count

            # _________________________
            # Typos
            if self.schema[c] in [np.str, np.object]:
                possible_typos = dirty_data[c].isin(list(set(self.distinct_values[c]).difference(
                    set(self.distinct_values_clean[c].keys()))))
                self.typos_mask[:, i] = possible_typos & (~swaps_mask[:, i])
                self.typos_values[c] = dirty_data[c].loc[self.typos_mask[:, i]].value_counts().to_dict()
                self.typos_num[c] = sum(self.typos_values[c].values())

                # _________________________
                # Estimate distincts typos number for each col
                if self.typos_num[c] > 0:
                    tmp_list = list(self.typos_values[c].values())
                    sum_tmp_list = sum(tmp_list)
                    self.distinct_typos_estimate[c] = max(
                        self._distinct_count_schlosser(
                            num_vals=len(tmp_list), freq_counts=tmp_list,
                            n_rows=(sum_tmp_list * scaling_factor), sample_size=sum_tmp_list),
                        self._distinct_count_haas_strokes(
                            num_vals=len(tmp_list), freq_counts=tmp_list,
                            n_rows=(sum_tmp_list * scaling_factor), sample_size=sum_tmp_list))
            else:
                self.typos_num[c] = 0

            # _________________________
            # Estimate distincts mv number for each col
            if self.mv_num[c] > 0:
                tmp_list = list(self.mv_values[c].values())
                sum_tmp_list = sum(tmp_list)
                self.distinct_mv_estimate[c] = max(
                    self._distinct_count_schlosser(
                        num_vals=len(tmp_list), freq_counts=tmp_list,
                        n_rows=(sum_tmp_list * scaling_factor), sample_size=sum_tmp_list),
                    self._distinct_count_haas_strokes(
                        num_vals=len(tmp_list), freq_counts=tmp_list,
                        n_rows=(sum_tmp_list * scaling_factor), sample_size=sum_tmp_list))

            touched_cells[c] |= self.typos_mask[:, i] | self.outliers_mask[c] | replacement

            i += 1

        print(f"Error dist count {list(touched_cells.sum())}")
        print(f"Original error count {self.num_errors_per_col}")
        t = touched_cells != self.initial_difference_mask
        assert (touched_cells.sum() <= self.num_errors_per_col).all(), \
            "More errors found in error distribution than differences between clean and dirty data."

        touched_cells = None

        # _________________________
        # Statistics clean
        self.clean_min = clean_data.min(numeric_only=True)
        self.clean_max = clean_data.max(numeric_only=True)
        self.clean_mean = clean_data.mean(numeric_only=True)
        self.clean_mode = clean_data.mode()

        self.clean_q25 = clean_data.quantile(q=0.25, numeric_only=True)
        self.clean_q50 = clean_data.quantile(q=0.5, numeric_only=True)
        self.clean_q75 = clean_data.quantile(q=0.75, numeric_only=True)

        self.clean_var = clean_data.var(numeric_only=True)
        self.clean_skew = clean_data.skew(numeric_only=True)
        self.clean_kurtosis = clean_data.kurtosis(numeric_only=True)

        # Statistics dirty
        dirty_data = dirty_data.fillna(0)
        self.dirty_min = dirty_data.min(numeric_only=True)
        self.dirty_max = dirty_data.max(numeric_only=True)
        self.dirty_mean = dirty_data.mean(numeric_only=True)
        self.dirty_mode = dirty_data.mode()

        self.dirty_q25 = dirty_data.quantile(q=0.25, numeric_only=True)
        self.dirty_q50 = dirty_data.quantile(q=0.5, numeric_only=True)
        self.dirty_q75 = dirty_data.quantile(q=0.75, numeric_only=True)

        self.dirty_var = dirty_data.var(numeric_only=True)
        self.dirty_skew = dirty_data.skew(numeric_only=True)
        self.dirty_kurtosis = dirty_data.kurtosis(numeric_only=True)

    def _distinct_count_schlosser(self, num_vals, freq_counts, n_rows, sample_size):
        """Peter J. Haas, Jeffrey F. Naughton, S. Seshadri, and Lynne Stokes. Sampling-Based Estimation of the Number of
         Distinct Values of an Attribute. VLDB'95, Section 3.2."""
        # Inverted frequency histogram
        max_count = max(freq_counts)

        freq_counts_inv = np.zeros(max_count)
        for f in freq_counts:
            if f != 0:
                freq_counts_inv[f - 1] += 1

        q = sample_size / n_rows
        one_minus_q = 1 - q

        numer_sum, denom_sum, i = 0.0, 0.0, 0
        i_plus_one = 1
        for f in freq_counts_inv:
            if f != 0:
                numer_sum += float(math.pow(one_minus_q, i_plus_one) * f)
                denom_sum += float(i_plus_one * q * math.pow(one_minus_q, i) * f)
            i += 1
            i_plus_one += 1
        if denom_sum == 0.0:
            return num_vals
        return int(round(num_vals + freq_counts_inv[0] * numer_sum / denom_sum))

    def _distinct_count_haas_strokes(self, num_vals, freq_counts, n_rows, sample_size):
        """Haas, Peter J., and Lynne Stokes. "Estimating the number of classes in a finite population." Journal of the
        American Statistical Association 93.444 (1998): 1475-1487."""
        HAAS_AND_STOKES_ALPHA1 = 0.9

        # Inverted frequency histogram
        max_count = max(freq_counts)

        freq_counts_inv = np.zeros(max_count)
        for f in freq_counts:
            if f != 0:
                freq_counts_inv[f-1] += 1

        # # remove zeroes
        # freq_counts_inv = np.delete(freq_counts_inv, np.where(freq_counts_inv == 0))

        q = sample_size / n_rows
        f1 = freq_counts_inv[0]

        # compute basic Duj1 estimate
        duj1 = self._getDuj1Estimate(q, f1, sample_size, num_vals)

        # compute gamma based on Duj1
        gamma = self._getGammaSquared(duj1, freq_counts_inv, sample_size, n_rows)
        d = -1

        if gamma < HAAS_AND_STOKES_ALPHA1:
            d = self._getDuj2Estimate(q, f1, sample_size, num_vals, gamma)

        else:
            d = self._getSh3Estimate(q, freq_counts_inv, num_vals)
        return int(d)

    def _getDuj1Estimate(self, q, f1, n, dn):
        return dn / (1 - ((1 - q) * f1) / n)

    def _getDuj2Estimate(self, q, f1, n, dn, gammaDuj1):
        return (dn - (1 - q) * f1 * math.log(1 - q) * gammaDuj1 / q) / (1 - ((1 - q) * f1) / n)

    def _getSh3Estimate(self, q, f, dn):
        fraq11, fraq12, fraq21, fraq22 = 0.0, 0.0, 0.0, 0.0
        for i in range(1, len(f)):
            if f[i - 1] != 0:
                fraq11 += i * q * q * math.pow(1 - q * q, i - 1) * f[i - 1]
                fraq12 += (math.pow(1 - q * q, i) - math.pow(1 - q, i)) * f[i - 1]
                fraq21 += math.pow(1 - q, i) * f[i - 1]
                fraq22 += i * q * math.pow(1 - q, i - 1) * f[i - 1]

        return dn + f[0] * fraq11 / fraq12 * math.pow(fraq21 / fraq22, 2)

    def _getGammaSquared(self, D, f, n, N):
        gamma = 0.0
        for i in range(1, len(f)):
            if f[i - 1] != 0:
                gamma += i * (i - 1) * f[i - 1]
            gamma *= D / n / n
            gamma += D / N - 1
        return max(gamma, 0)

    def __str__(self):
        clean_dict = {k: len(self.distinct_values_clean[k]) for k in self.distinct_values_clean.keys()}
        dirty_swaps = sum([sum(k_v.values()) for k_v in self.swaps_dict_nums.values()]) + sum([sum(k_v.values()) for k_v in self.swaps_dict_with_str.values()])
        str_obj = f"Rows\t {self.original_rows}\n" \
                f"Cols\t {self.original_cols}\n" \
                f"Clean min\t {self.clean_min}\n" \
                f"Clean max\t {self.clean_max}\n" \
                f"Clean mean\t {self.clean_mean}\n" \
                f"Clean q25\t {self.clean_q25}\n" \
                f"Clean q50\t {self.clean_q50}\n" \
                f"Clean q75\t {self.clean_q75}\n" \
                f"Clean var\t {self.clean_var}\n" \
                f"Clean kurt\t {self.clean_kurtosis}\n" \
                f"Clean skew\t {self.clean_skew}\n\n" \
                f"Dirty min\t {self.dirty_min}\n" \
                f"Dirty max\t {self.dirty_max}\n" \
                f"Dirty mean\t {self.dirty_mean}\n" \
                f"Dirty q25\t {self.dirty_q50}\n" \
                f"Dirty q50\t {self.dirty_q50}\n" \
                f"Dirty q75\t {self.dirty_q75}\n" \
                f"Dirty var\t {self.dirty_var}\n" \
                f"Dirty kurt\t {self.dirty_kurtosis}\n" \
                f"Dirty skew\t {self.dirty_skew}\n\n" \
                f"Clean distincts\t {clean_dict}\n" \
                f"Dirty distincts\t {self.distinct_num}\n" \
                f"Dirty mv\t {self.mv_num}\n" \
                f"Dirty outliers\t {self.outliers_num}\n" \
                f"Dirty typos\t {self.typos_num}\n" \
                f"Dirty swaps\t {dirty_swaps}\n" \
                f"Dirty replacements\t {self.distinct_replacements_count}\n"

        return str_obj
