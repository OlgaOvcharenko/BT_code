import math

import numpy as np

from error_distribution import ErrorDistribution


def compare_error_distributions_statistics(cols, original: ErrorDistribution, generated: ErrorDistribution):
    print_statistics(cols=cols, original=original, generated=generated)
    compare_distincts(cols=cols, original=original, generated=generated)


def compare_error_distributions_statistics_sp(cols, original: ErrorDistribution, generated: ErrorDistribution):
    print_statistics_sp(cols=cols, original=original, generated=generated)
    compare_distincts_sp(cols=cols, original=original, generated=generated)


def compare_distincts(cols, original: ErrorDistribution, generated: ErrorDistribution):
    i = 0
    for c in cols:
        print("--------------------------------------------------------")
        print(c)

        dist_dirty_num = original.distinct_num[c]
        dist_clean_num = len(original.distinct_values_clean[c])
        dist_estimate = original.distincts_estimate[c]
        ratio = math.ceil(dist_clean_num / dist_dirty_num)
        ratio_add_distincts = round(original.original_rows * original.scaling_factor) * ratio
        distincts_new_num = dist_estimate if dist_estimate > 0 else ratio_add_distincts
        expected_dist = dist_clean_num + (original.distinct_typos_estimate.get(c, 0) + original.distinct_mv_estimate.get(c, 0) +
                                          len(original.outliers_values.get(c, 0)) * original.scaling_factor)

        is_within_disincts = (original.distincts_estimate[c] - original.distinct_num[c]) * 0.1 < \
                             generated.distinct_num[c] < \
                             (original.distincts_estimate[c] + original.distinct_num[c]) * 0.1
        # assert not is_within_disincts, "New number of distinct values is not within 10% of estimated."

        print(f"Distinct original dirty: \t {original.distinct_num[c]}")
        print(f"Distinct original clean: \t {dist_clean_num}")
        print(f"Distinct estimated dirty (whole dataset): \t {original.distincts_estimate[c]}, "
              f"distinct with ratio \t {distincts_new_num}")
        print(f"Distinct estimated dirty (replicate + errors): \t {expected_dist}")
        print(f"Distinct generated dirty: \t {generated.distinct_num[c]}\n")

        # Swaps
        if original.swaps_col_unique_count.get(c, 0) > 0:
            print(f"Swaps original dirty: \t {original.swaps_col_count[i]}")
            print(f"Swaps generated dirty: \t {generated.swaps_col_count[0, i]}\n")

            print(f"Unique swaps count - original dirty: \t {original.swaps_col_unique_count[c]}")
            print(f"Unique swaps count - generated dirty: \t {generated.swaps_col_unique_count[c]}\n")
            assert round(original.swaps_col_count[i] * original.scaling_factor) == \
                   generated.swaps_col_count[0, i], \
                f"In {c} changed swaps number, original - {round(original.swaps_col_count[i] * original.scaling_factor)}, " \
                f"gen - {generated.swaps_col_count[i]}"

        # Replacements
        if original.distinct_replacements_count.get(c, 0) > 0:
            print(f"Replacements original dirty: \t {original.distinct_replacements_count[c]}")
            print(f"Replacements generated dirty: \t {generated.distinct_replacements_count[c]}\n")
            assert round(original.distinct_replacements_count[c] * original.scaling_factor * 0.9) <= \
                   generated.distinct_replacements_count[c] <= \
            round(original.distinct_replacements_count[c] * original.scaling_factor * 1.1), \
                f"In {c} changed replacements distribution, original - " \
                f"{round(original.distinct_replacements_count[c] * original.scaling_factor)}, " \
                f"gen - {generated.distinct_replacements_count[c]}"

            print(f"Unique replacements count - original dirty: \t {len(original.replacements_dicts[c])}")
            print(f"Unique replacements count - generated dirty: \t {len(generated.replacements_dicts[c])} \n")
            assert len(original.replacements_dicts[c]) == len(generated.replacements_dicts[c]), \
                f"In {c} changed replacements distribution, original - {len(original.replacements_dicts[c])}, " \
                f"gen - {len(generated.replacements_dicts[c])}"

        # MV
        if original.mv_num.get(c, 0) > 0:
            print(f"MV original dirty: \t {original.mv_num[c]}")
            print(f"MV generated dirty: \t {generated.mv_num[c]}\n")

            print(f"Distinct estimated mv: \t {original.distinct_mv_estimate[c]}")
            print(f"Unique MV count - original dirty: \t {len(original.mv_values[c])}")
            print(f"Unique MV count - generated dirty: \t {len(generated.mv_values[c])}\n")
            assert len(original.mv_values[c]) <= len(generated.mv_values[c]) or \
                   (original.mv_num[c] <= generated.mv_num[c]), \
                f"In {c} generated less mv {generated.mv_num[c]} than in dirty dataset {original.mv_num[c]}"

        # Outliers
        if original.outliers_num.get(c, 0) > 0:
            print(f"Outliers original dirty: \t {original.outliers_num[c]}")
            print(f"Outliers generated dirty: \t {generated.outliers_num[c]}\n")

            print(f"Unique outliers count - original dirty: \t {len(original.outliers_values[c])}")
            print(f"Unique outliers count - generated dirty: \t {len(generated.outliers_values[c])}\n")
            assert len(original.outliers_values[c]) <= len(generated.outliers_values[c]) or \
                   (original.outliers_num[c] <= generated.outliers_num[c]), \
                f"Generated less outliers {original.outliers_num[c]} " \
                f"than in dirty dataset {generated.outliers_num[c]}"

        # Typos
        if original.typos_num.get(c, 0) > 0:
            print(f"Typos original dirty: \t {original.typos_num[c]}")
            print(f"Typos generated dirty: \t {generated.typos_num[c]}\n")

            print(f"Distinct estimated typo: \t {original.distinct_typos_estimate[c]}")
            print(f"Unique typos count - original dirty: \t {len(original.typos_values[c])}")
            print(f"Unique typos count - generated dirty: \t {len(generated.typos_values[c])}\n")
            assert len(original.typos_values[c]) <= len(generated.typos_values[c]) or \
                   (original.typos_num[c] <= generated.typos_num[c]), \
                f"Generated less typos {original.typos_num[c]} " \
                f"than in dirty dataset {generated.typos_num[c]}"

        # Number of introduced errors in general
        i += 1

    # Swaps
    original_swaps_num = sum([sum(d.values()) for d in original.swaps_dict_nums.values()])
    if original_swaps_num > 0:
        generated_swaps_num = sum([sum(d.values()) for d in generated.swaps_dict_nums.values()])
        print(f"Numerical swaps original dirty: \t {original_swaps_num}")
        print(f"Numerical swaps generated dirty: \t {generated_swaps_num}\n")
        assert round(original_swaps_num * original.scaling_factor) == \
               generated_swaps_num, \
            f"Changed numerical swaps number, original - {original_swaps_num}, " \
            f"gen - {generated_swaps_num}"

    original_swaps_str = sum([sum(d.values()) for d in original.swaps_dict_with_str.values()])
    if original_swaps_str > 0:
        generated_swaps_str = sum([sum(d.values()) for d in generated.swaps_dict_with_str.values()])
        print(f"Numerical swaps original dirty: \t {original_swaps_str}")
        print(f"Numerical swaps generated dirty: \t {generated_swaps_str}\n")
        assert round(original_swaps_str * original.scaling_factor) == \
               generated_swaps_str, \
            f"Changed numerical swaps number, original - {original_swaps_str}, " \
            f"gen - {generated_swaps_str}"


def print_statistics(cols, original: ErrorDistribution, generated: ErrorDistribution):
    for c in cols:
        if original.dirty_mean.get(c):
            print("--------------------------------------------------------")
            print(c)
            print(f"Mean original dirty: \t {original.dirty_mean[c]}")
            print(f"Mean generated dirty: \t {generated.dirty_mean[c]}\n")

            if (original.dirty_mean[c] - (original.dirty_mean[c] * 0.2) < generated.dirty_mean[c] <
                original.dirty_mean[c] + (original.dirty_mean[c] * 0.2)) or np.isnan(generated.dirty_mean[c]):
                f"INFO: In {c} mean differs by more than 20%"

            print(f"Var original dirty: \t {original.dirty_var[c]}")
            print(f"Var generated dirty: \t {generated.dirty_var[c]}\n")

            print(f"Q25 original dirty: \t {original.dirty_q25[c]}")
            print(f"Q25 generated dirty: \t {generated.dirty_q25[c]}\n")

            if original.dirty_q25[c] != generated.dirty_q25[c]:
                print("INFO: Q25 differs from original dirty data\n")

            print(f"Q50 original dirty: \t {original.dirty_q50[c]}")
            print(f"Q50 generated dirty: \t {generated.dirty_q50[c]}\n")

            if original.dirty_q50[c] != generated.dirty_q50[c]:
                print("INFO: Q50 differs from original dirty data\n")

            print(f"Q75 original dirty: \t {original.dirty_q75[c]}")
            print(f"Q75 generated dirty: \t {generated.dirty_q75[c]}\n")

            if original.dirty_q75[c] != generated.dirty_q25[c]:
                print("INFO: Q75 differs from original dirty data\n")

            iqr_original, iqr_gen = original.dirty_q75[c] - original.dirty_q25[c], \
                                    generated.dirty_q75[c] - generated.dirty_q25[c]

            if not (original.dirty_q25[c] - (iqr_original * 1.5) < generated.dirty_q25[c]
                    and generated.dirty_q75[c] < original.dirty_q75[c] + (iqr_original * 1.5)):
                print(f"In {c} faulty IQR")

            print(f"Skew original dirty: \t {original.dirty_skew[c]}")
            print(f"Skew generated dirty: \t {generated.dirty_skew[c]}\n")

            print(f"Kurtosis original dirty: \t {original.dirty_kurtosis[c]}")
            print(f"Kurtosis generated dirty: \t {generated.dirty_kurtosis[c]}\n")

            print(f"Min original dirty: \t {original.dirty_min[c]}")
            print(f"Min generated dirty: \t {generated.dirty_min[c]}\n")

            print(f"Max original dirty: \t {original.dirty_max[c]}")
            print(f"Max generated dirty: \t {generated.dirty_max[c]}\n")
            print("--------------------------------------------------------")


def print_statistics_sp(cols, original: ErrorDistribution, generated: ErrorDistribution):
    for c in cols:
        if original.dirty_mean.get(c):
            print("--------------------------------------------------------")
            print(c)
            print(f"Mean original dirty: \t {original.dirty_mean[c]}")
            print(f"Mean generated dirty: \t {generated.dirty_mean[c]}\n")

            if (original.dirty_mean[c] - (original.dirty_mean[c] * 0.2) < generated.dirty_mean[c] <
                   original.dirty_mean[c] + (original.dirty_mean[c] * 0.2)) or np.isnan(generated.dirty_mean[c]):
                f"INFO: In {c} mean differs by more than 20%"

            print(f"Var original dirty: \t {original.dirty_var[c]}")
            print(f"Var generated dirty: \t {generated.dirty_var[c]}\n")

            # print(f"Q25 original dirty: \t {original.dirty_q25[c]}")
            # print(f"Q25 generated dirty: \t {generated.dirty_q25[c]}\n")
            #
            # if original.dirty_q25[c] != generated.dirty_q25[c]:
            #     print("INFO: Q25 differs from original dirty data\n")
            #
            # print(f"Q50 original dirty: \t {original.dirty_q50[c]}")
            # print(f"Q50 generated dirty: \t {generated.dirty_q50[c]}\n")
            #
            # if original.dirty_q50[c] != generated.dirty_q50[c]:
            #     print("INFO: Q50 differs from original dirty data\n")
            #
            # print(f"Q75 original dirty: \t {original.dirty_q75[c]}")
            # print(f"Q75 generated dirty: \t {generated.dirty_q75[c]}\n")
            #
            # if original.dirty_q75[c] != generated.dirty_q25[c]:
            #     print("INFO: Q75 differs from original dirty data\n")
            #
            # iqr_original, iqr_gen = original.dirty_q75[c] - original.dirty_q25[c], \
            #                         generated.dirty_q75[c] - generated.dirty_q25[c]
            #
            # if not (original.dirty_q25[c] - (iqr_original * 1.5) < generated.dirty_q25[c]
            #         and generated.dirty_q75[c] < original.dirty_q75[c] + (iqr_original * 1.5)):
            #     print(f"In {c} faulty IQR")

            print(f"Min original dirty: \t {original.dirty_min[c]}")
            print(f"Min generated dirty: \t {generated.dirty_min[c]}\n")

            print(f"Max original dirty: \t {original.dirty_max[c]}")
            print(f"Max generated dirty: \t {generated.dirty_max[c]}\n")
            print("--------------------------------------------------------")


def compare_distincts_sp(cols, original: ErrorDistribution, generated: ErrorDistribution):
    i = 0
    for c in cols:
        print("--------------------------------------------------------")
        print(c)

        dist_dirty_num = original.distinct_num[c]
        dist_clean_num = len(original.distinct_values_clean[c])
        dist_estimate = original.distincts_estimate[c]
        ratio = math.ceil(dist_clean_num / dist_dirty_num)
        ratio_add_distincts = round(original.original_rows * original.scaling_factor) * ratio
        distincts_new_num = dist_estimate if dist_estimate > 0 else ratio_add_distincts
        expected_dist = dist_clean_num + (
                    original.distinct_typos_estimate.get(c, 0) + original.distinct_mv_estimate.get(c, 0) +
                    len(original.outliers_values.get(c, 0)) * original.scaling_factor)

        is_within_disincts = (original.distincts_estimate[c] - original.distinct_num[c]) * 0.1 < \
                             generated.distinct_num[c] < \
                             (original.distincts_estimate[c] + original.distinct_num[c]) * 0.1
        # assert not is_within_disincts, "New number of distinct values is not within 10% of estimated."

        print(f"Distinct original dirty: \t {original.distinct_num[c]}")
        print(f"Distinct original clean: \t {dist_clean_num}")
        print(f"Distinct estimated dirty (whole dataset): \t {original.distincts_estimate[c]}, "
              f"distinct with ratio \t {distincts_new_num}")
        print(f"Distinct estimated dirty (replicate + errors): \t {expected_dist}")
        print(f"Distinct generated dirty: \t {generated.distinct_num[c]}\n")

        # Replacements FIXME
        if original.distinct_replacements_count.get(c, 0) > 0:
            print(f"Replacements original dirty: \t {original.distinct_replacements_count[c]}")
            print(f"Replacements generated dirty: \t {generated.distinct_replacements_count[c]}\n")
            assert round(original.distinct_replacements_count[c] * original.scaling_factor) * 0.9 <= \
                   generated.distinct_replacements_count[c] <= round(original.distinct_replacements_count[c] * original.scaling_factor) * 1.1, \
                f"In {c} changed replacements distribution, original - " \
                f"{round(original.distinct_replacements_count[c] * original.scaling_factor)}, " \
                f"gen - {generated.distinct_replacements_count[c]}"

        # MV
        if original.mv_num.get(c, 0) > 0:
            print(f"MV original dirty: \t {original.mv_num[c]}")
            print(f"MV generated dirty: \t {generated.mv_num[c]}\n")

            print(f"Distinct estimated mv: \t {original.distinct_mv_estimate[c]}")
            print(f"Unique MV count - original dirty: \t {len(original.mv_values[c])}")
            print(f"Unique MV count - generated dirty: \t {len(generated.mv_values[c])}\n")
            assert len(original.mv_values[c]) <= len(generated.mv_values[c]) or \
                   (original.mv_num[c] <= generated.mv_num[c]), \
                f"In {c} generated less mv {generated.mv_num[c]} than in dirty dataset {original.mv_num[c]}"

        # Outliers
        if original.outliers_num.get(c, 0) > 0:
            print(f"Outliers original dirty: \t {original.outliers_num[c]}")
            print(f"Outliers generated dirty: \t {generated.outliers_num[c]}\n")

            print(f"Unique outliers count - original dirty: \t {len(original.outliers_values[c])}")
            print(f"Unique outliers count - generated dirty: \t {len(generated.outliers_values[c])}\n")
            assert len(original.outliers_values[c]) <= len(generated.outliers_values[c]) or \
                   (original.outliers_num[c] <= generated.outliers_num[c]), \
                f"Generated less outliers {original.outliers_num[c]} " \
                f"than in dirty dataset {generated.outliers_num[c]}"

        # Typos
        if original.typos_num.get(c, 0) > 0:
            print(f"Typos original dirty: \t {original.typos_num[c]}")
            print(f"Typos generated dirty: \t {generated.typos_num[c]}\n")

            print(f"Distinct estimated typo: \t {original.distinct_typos_estimate[c]}")
            print(f"Unique typos count - original dirty: \t {len(original.typos_values[c])}")
            print(f"Unique typos count - generated dirty: \t {len(generated.typos_values[c])}\n")
            assert len(original.typos_values[c]) <= len(generated.typos_values[c]) or \
                   (original.typos_num[c] <= generated.typos_num[c]), \
                f"Generated less typos {original.typos_num[c]} " \
                f"than in dirty dataset {generated.typos_num[c]}"

        # Number of introduced errors in general
        i += 1

    # Swaps
    original_swaps_num = sum([sum(d.values()) for d in original.swaps_dict_nums.values()])
    if original_swaps_num > 0:
        generated_swaps_num = sum([sum(d.values()) for d in generated.swaps_dict_nums.values()])
        print(f"Numerical swaps original dirty: \t {original_swaps_num}")
        print(f"Numerical swaps generated dirty: \t {generated_swaps_num}\n")
        # assert round(original_swaps_num * original.scaling_factor) * 0.9 <= \
        #        generated_swaps_num <= round(original_swaps_num * original.scaling_factor) * 1.1, \
        #     f"Changed numerical swaps number, original - {original_swaps_num}, " \
        #     f"gen - {generated_swaps_num}"

        if not (round(original_swaps_num * original.scaling_factor) * 0.9 <= \
               generated_swaps_num <= round(original_swaps_num * original.scaling_factor) * 1.1):
            print(f"Changed numerical swaps number, original - {original_swaps_num}, " \
            f"gen - {generated_swaps_num}")

    original_swaps_str = sum([sum(d.values()) for d in original.swaps_dict_with_str.values()])
    if original_swaps_str > 0:
        generated_swaps_str = sum([sum(d.values()) for d in generated.swaps_dict_with_str.values()])
        print(f"Numerical swaps original dirty: \t {original_swaps_str}")
        print(f"Numerical swaps generated dirty: \t {generated_swaps_str}\n")
        # assert round(original_swaps_str * original.scaling_factor) * 0.9 <= \
        #        generated_swaps_str <= round(original_swaps_str * original.scaling_factor) * 1.1, \
        #     f"Changed numerical swaps number, original - {original_swaps_str}, " \
        #     f"gen - {generated_swaps_str}"

        if not (round(original_swaps_str * original.scaling_factor) * 0.9 <= \
               generated_swaps_str <= round(original_swaps_str * original.scaling_factor) * 1.1):
            print(f"Changed numerical swaps number, original - {original_swaps_str}, " \
            f"gen - {generated_swaps_str}")
