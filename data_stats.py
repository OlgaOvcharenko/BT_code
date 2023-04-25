import sys

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from scipy import stats

import utils
from utils import Utils

MISSING_VALUES = [-9999, np.NAN, np.inf, sys.float_info.max, sys.float_info.min]


class DataStats:
    """Stats before / after generation"""

    def __init__(self):
        self.shape = None

        self.distinct = None
        self.FDs = None
        self.outliers = None
        self.missing_values = None

        self.min = None
        self.max = None
        self.mean = None
        self.var = None
        self.skew = None
        self.kurtosis = None

        self.mode = None

        self.errors_count_per_column = {}

    def set_FDs(self, FDs):
        self.FDs = FDs

    def set_outliers(self, outliers):
        self.outliers = outliers

    def set_missing_values(self, missing_values):
        self.missing_values = missing_values

    def compute_stats(self, data, fds_path: str = None):
        self.shape = data.shape

        self.distinct = [Utils.get_distinct_values(data, col_i) for col_i in range(0, data.shape[1])]
        self.FDs = Utils.discoverFD(data, 0.8) if not fds_path else utils.parse_fds_from_file(fds_path, self.shape[1])  # FIXME

        # Outliers mask
        outliers_per_col = [np.reshape(Utils.detect_col_outliers_by_IQR(data.values[:, col_i], False)[0], (data.shape[0], 1))
                                        if not isinstance(data.iat[0, col_i], str) else np.zeros((data.shape[0], 1))
                                        for col_i in range(0, data.shape[1])]
        self.outliers = np.concatenate(outliers_per_col, axis=1)

        # Missing values mask
        self.missing_values = np.full(data.shape, True, dtype=bool)
        for value in MISSING_VALUES:
            self.missing_values = self.missing_values & np.ma.getmask(np.ma.masked_values(data, value))

        self.min, self.max = np.min(data, axis=0), np.max(data, axis=0)

        means, variances, skews, kurtosises = [], [], [], []
        for i in data:
            if is_numeric_dtype(data[i]):
                means.append(np.mean(data[i]))
                variances.append(np.var(data[i]))
                skews.append(stats.skew(data[i]))
                kurtosises.append(stats.kurtosis(data[i]))

            else:
                means.append(0.0)
                skews.append(0.0)
                variances.append(0.0)
                kurtosises.append(0.0)

            self.errors_count_per_column[i] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}

        self.skew, self.var, self.mean, self.kurtosis = np.array(skews), np.array(variances), np.array(means), np.array(kurtosises)

    def compare_stats_after_generation(self, new_stats, errors_introduced: bool = False):
        if isinstance(new_stats, DataStats):
            # for d1, d2 in zip(self.distinct, new_stats.distinct):
            #     if d1 != d2:
            #         print(d1)
            #         print(d2)

            if not errors_introduced:
                assert all([(d1 == d2).all() if not isinstance(d1 == d2, bool) else d1 == d2 for d1, d2 in zip(self.distinct, new_stats.distinct)]), \
                    'Distinct values differ'
                assert (self.min == new_stats.min).all(), 'Mins differ'  # FIXME not true
                assert (self.max == new_stats.max).all(), 'Maxs differ'  # FIXME not true

            # Check skewness
            for col_i in range(0, self.shape[1]):
                # Highly skewed
                if self.skew[col_i] < -1:
                    assert (new_stats.skew[col_i] < -0.8)
                elif self.skew[col_i] > 1:
                    assert (new_stats.skew[col_i] > 0.5)

                # Moderately skewed
                elif -1 < self.skew[col_i] < -0.5:
                    assert (-1.2 < new_stats.skew[col_i] < -0.3), 'Skew difference is more than 20%'
                elif 0.5 < self.skew[col_i] < 1:
                    assert (0.3 < new_stats.skew[col_i] < 1.2), 'Skew difference is more than 20%'

                # Approximately symmetric
                elif -0.5 < self.skew[col_i] < 0.5:
                    assert (-0.7 < new_stats.skew[col_i] < 0.7), 'Skew difference is more than 20%'   # FIXME is valid check

                else:
                    pass

            # Compare variance (differ within 20%)
            assert all(self.var * -0.2 <= abs(self.var - new_stats.var)) & all(abs(self.var - new_stats.var) <= self.var * 0.2), \
                'Variances differ by more than 20 %'

        else:
            raise Exception('compare_stats: both stats objects should be instance of DataStats')
