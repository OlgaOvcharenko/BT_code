import functools
import random
import statistics
import typing
from typing import List

import numpy as np
import pandas as pd
import pyspark
from scipy import stats

import error_distribution
import static_utils


def isFD(X, Y, ndX):
    ctab = pd.crosstab(X, Y, rownames=['X'], colnames=['Y'])
    row_sums = np.sum(ctab != 0, axis=1)
    A_det_B = np.sum(row_sums == 1) == ndX
    ratio = np.sum(np.max(ctab, axis=1)) / X.shape[0]
    return A_det_B, ratio


def levenshtein(a, b):
    if not a:
        return len(b)
    if not b:
        return len(a)
    return min(levenshtein(a[1:], b[1:])+(a[0] != b[0]),
               levenshtein(a[1:], b)+1,
               levenshtein(a, b[1:])+1)


def jaccard_similarity(a: List[str], b: List[str]):
    a_set, b_set = set(a), set(b)

    nominator, denominator = a_set.intersection(b_set), a_set.union(b_set)
    return len(nominator) / len(denominator)


def convert_str_to_list(val_str):
    l = []
    l[:0] = val_str
    return l


class Utils:
    @staticmethod
    def get_distinct_values(data, col):
        values = list(set(np.unique(data.iloc[:, col])))
        if isinstance(values[0], int) or isinstance(values[0], float):
            values = list(np.array(values)[np.logical_not(np.isnan(values))])
        return values

    @staticmethod
    def EDM(A, B, is_str: bool = False):
        if is_str:
            dist_matrix = np.zeroes((1, B.shape[0]))
            for i in range(0, B.shape[0]):
                dist_matrix[0, i] = jaccard_similarity(convert_str_to_list(A), convert_str_to_list(B.loc[i, 0]))
            return dist_matrix

        else:
            p1 = np.sum(A**2, axis=1)[:, np.newaxis]
            p2 = np.sum(B**2, axis=1)
            p3 = -2 * np.dot(A, B.T)
            return np.round(np.sqrt(p1 + p2 + p3), 2)

    @staticmethod
    def EDM(X):
        G = np.matmul(X, X.T)
        Y = -2 * G + (np.diag(G) * np.identity(G.shape[0])) + (np.identity(G.shape[0]) * np.diag(G))
        return Y

    @staticmethod
    def find_nearest(array, value):
        array = np.asarray(array)
        dist = (np.abs(array - value))
        dist[dist == 0] = max(dist)
        return array[dist.argmin()]

    @staticmethod
    def discoverFD(X, threshold: float):
        assert threshold <= 1 or threshold >= 0, 'discoverFD: Threshold required in interval [0, 1]'

        n, d = X.shape[0], X.shape[1]
        FD = np.diag(np.ones((d, 1)))
        cm = np.zeros((1, d))
        for i in range(0, d):
            cm[0, i] = np.sum(pd.crosstab(X.iloc[:, i], 1)) != 0

        FD = FD + (cm == 1)
        FD = FD + (cm.T == n)
        FD = FD != 0

        cm2 = np.argsort(cm.transpose())
        for i in range(0, d):
            index = cm2[i, 0]
            ndX = cm[0, index]

            if ndX != 0 and ndX != n-1:
                X_i = X.iloc[:, index]
                k = 0 if threshold < 1 else i

                for j in range(k, d):
                    if j != i and j > 0 and j <= d:
                        index_j = cm2[j, 0]
                        A_det_B, ratio = isFD(X_i, X.iloc[:, index_j], ndX)
                        if A_det_B or ratio >= threshold:
                            FD[index, index_j] = ratio

        return FD

    @staticmethod
    def detect_col_outliers_from_dist(col_dirty, col_clean, replace_outliers: False):
        lower_limit, upper_limit = col_clean.mean() - 3 * col_clean.std(), col_clean.mean() + 3 * col_clean.std()
        mask = (upper_limit < col_dirty) | (col_dirty < lower_limit)

        if replace_outliers:
            # Capping outliers
            new_X = np.where(col_dirty > upper_limit, upper_limit, np.where(col_dirty < lower_limit, lower_limit, col_dirty))
            return mask, new_X, lower_limit, upper_limit
        else:
            return mask, lower_limit, upper_limit

    @staticmethod
    def detect_col_outliers_by_IQR(col_dirty, col_clean, replace_outliers: False):
        q1 = col_clean.quantile(0.25)
        q3 = col_clean.quantile(0.75)
        iqr = q3 - q1
        lower_limit, upper_limit = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (upper_limit < col_dirty) | (col_dirty < lower_limit)

        if replace_outliers:
            # Capping outliers
            new_X = np.where(col_dirty > upper_limit, upper_limit,
                             np.where(col_dirty < lower_limit, lower_limit, col_dirty))
            return mask, new_X, lower_limit, upper_limit

        else:
            return mask, lower_limit, upper_limit


    @staticmethod
    def impute_col_by_mean(col):
        col[np.isnan(col)] = np.mean(col, axis=0)
        return col

    @staticmethod
    def impute_col_by_mode(col):
        col[np.isnan(col)] = stats.mode(col)
        return col

    @staticmethod
    def impute_col_by_median(col):
        col[np.isnan(col)] = np.median(col, axis=0)
        return col

    @staticmethod
    def impute_col_by_FD(col):
        assert False, 'Not implemented'

    @staticmethod
    def detect_schema_from_rows_sample(data: pd.DataFrame):
        data_schema = {}
        data_schema_parse = {}
        # random_indices = np.random.choice(data.shape[0], size=int(data.shape[0] * 0.1), replace=False)

        random_indices = range(0, 30)
        for col in data.columns:
            instances = []  # List with all instances to take into account MV
            for row_ix in random_indices:
                value = data[col].iloc[row_ix]

                if value is not None:
                    if is_float(value):
                        if is_int(value):
                            if float(value) == int(value):
                                instances.append(np.int)
                            else:
                                instances.append(np.float)
                        else:
                            instances.append(np.float)

                    elif value.isdigit():
                        instances.append(np.int)

                    elif isinstance(value, bool):
                        instances.append(np.bool)

                    elif isinstance(value, str):
                        instances.append(np.str)

            data_schema[col] = statistics.mode(instances)
            data_schema_parse[col] = np.float if data_schema[col] == np.int else data_schema[col]
            # if data_schema_parse[col] == np.int:
            #     if not no_missing_values(data[col]):
            #         data_schema_parse[col] = np.float

        return data_schema, data_schema_parse

    @staticmethod
    def detect_schema_from_df(df):
        data_schema = []
        str_mask = []

        for col_type in df.dtypes.values:
            if col_type is np.dtype('float64') or col_type is np.dtype('float32'):
                data_schema.append('float')

            elif col_type is np.dtype('int64') or col_type is np.dtype('int32'):
                data_schema.append('int')

            elif col_type is np.dtype('bool'):
                data_schema.append('bool')

            elif isinstance(col_type, object):
                data_schema.append('str')

        return data_schema


def is_float(num: str):
    try:
        float(num)
        return not (np.isnan(float(num)) or np.isinf(float(num)) or pd.isna(num))
    except ValueError:
        return False


def float_or_str(val):
    return float(val) if is_float(val) else str(val)


def is_int(num: str):
    try:
        int(num)
        return True
    except ValueError:
        return False


def no_missing_values(col):
    try:
        flag = not (col.values == error_distribution.MISSING_VALUES)
        col.astype(int)
        return flag

    except ValueError:
        return False


def parse_fds_from_file(file_name: str, cols: int):
    """Parse file that contains col ids as dependencies like 0 -> 2,3"""

    with open(file_name) as f:
        lines = f.readlines()

    FD = np.identity(cols)
    for line in lines:
        parts = line.replace(' ', '').replace('\n', '').split('->')
        assert len(parts) == 2

        left, right = [int(c) for c in parts[0].split(',')], [int(c) for c in parts[1].split(',')]
        for l in left:
            for r in right:
                FD[l, r] = 1
    return FD


def col_means(data: pd.DataFrame, mask):
    col_means(data, mask, data.columns)


def col_means(data: pd.DataFrame, mask, selected_cols, clean_mean):
    # for col in selected_cols:
    #     for row_ix in data.row
    #     data[col]
    if len(selected_cols) == 0:
        return

    i = 0
    means = {col: 0 for col in selected_cols}
    for c_name in selected_cols:
        # means[c_name] = data[c_name].loc[mask[:, data.columns.get_loc(c_name)]].astype(float).sum() / data.shape[0]
        tmp_fr = pd.to_numeric(data[c_name].loc[mask[:, i]], errors='coerce').fillna(0)
        tmp_mean = np.average(tmp_fr, weights=np.ones_like(tmp_fr) / tmp_fr.shape[0])
        # print(tmp_mean)
        means[c_name] = tmp_mean if not (np.isnan(tmp_mean) or np.isinf(tmp_mean)) else clean_mean[c_name]

        # means[c_name] = pd.to_numeric(data[c_name].loc[mask[:, i]], errors='coerce').fillna(0).mean()

        i += 1
    # print(means)
    return means


def col_means_sp(data: pd.DataFrame, selected_cols):
    # for col in selected_cols:
    #     for row_ix in data.row
    #     data[col]
    if len(selected_cols) == 0:
        return

    means = {col: 0 for col in selected_cols}
    for c_name in selected_cols:
        # means[c_name] = data[c_name].loc[mask[:, data.columns.get_loc(c_name)]].astype(float).sum() / data.shape[0]
        means[c_name] = data[c_name].fillna(0).mean(numeric_only=False)
    return means


def write_err_dist(err_dist, path: str, is_cluster: bool):
    content = err_dist.__str__()

    if static_utils.is_in_cluster() and is_cluster:
        hdfs_client = static_utils.write_to_hdfs()
        with hdfs_client.write(path, encoding="utf-8") as file_errors:
            file_errors.write(content)
    else:
        with open(path, "w") as file_errors:
            file_errors.write(content)


def merge(f, l, h):
    if (h - l < 3):
        return functools.reduce(pyspark.sql.DataFrame.union, f[l:h])
    else:
        m = l + (h - l) // 2

        lr = merge(f, l, m)
        hr = merge(f, m, h)
        r = lr.union(hr)
        return r


def rand_except(nrows: int, sample_size: int, except_list: typing.List[int]):
    if not except_list:
        return random.sample(range(nrows), sample_size)

    if  (len(except_list) + sample_size) > nrows:
        print("WARNING: random_except: some indices will be overwritten")
        return random.sample(range(nrows), sample_size)

    except_list.sort()

    res = np.zeros(sample_size)
    i, j, current_row = 0, 0, 0

    while i < sample_size and j < len (except_list):
        if current_row == except_list[j]:
            j += 1

        else:
            res[i] = current_row
            i += 1

        current_row += 1

    while i < sample_size:
        res[i] = current_row
        i += 1
        current_row += 1

    probability = float(sample_size) / nrows
    while current_row < nrows and j < len(except_list):
        if current_row == except_list[j]:
            j += 1

        elif random.random() < probability:
                res[random.randint(0, sample_size-1)] = current_row

        current_row += 1

    while current_row < nrows:
        if random.random() < probability:
            res[random.randint(0, sample_size - 1)] = current_row

        current_row += 1

    return list(res)
