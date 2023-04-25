import numpy as np
import pandas as pd
from parameterized import parameterized

import generator as gen
import univar_stats


def scale_both_and_rand_transform_dirty(data_path_clean: str, data_path_dirty: str, is_header: int, scale_factor: float):
    data_clean = pd.read_csv(data_path_clean, header=is_header, sep=',')
    _, new_data_clean = gen.run(data_clean, gen.ExecMode.CP, scale_factor=scale_factor,
                          generate_rand_transformations=False)
    # univar_stats.Stats.get_stats(data_clean, new_data_clean)

    data_dirty = pd.read_csv(data_path_dirty, header=is_header, sep=',')
    _, new_data_dirty = gen.run(data_dirty, gen.ExecMode.CP, scale_factor=scale_factor,
                                generate_rand_transformations=True)
    # univar_stats.Stats.get_stats(data_dirty, new_data_dirty)

    print("Original stats:")
    univar_stats.Stats.get_stats(data_clean, data_dirty)
    print("\nScaled stats after random transformations:")
    univar_stats.Stats.get_stats(new_data_clean, new_data_dirty)


def scale_both(data_path_clean: str, data_path_dirty: str, is_header: int, scale_factor: float):
    data_clean = pd.read_csv(data_path_clean, header=is_header, sep=',')
    _, new_data_clean = gen.run(data_clean.values, gen.ExecMode.CP, scale_factor=scale_factor,
                                generate_rand_transformations=False)
    # univar_stats.Stats.get_stats(data_clean, new_data_clean)

    data_dirty = pd.read_csv(data_path_dirty, header=is_header, sep=',')
    _, new_data_dirty = gen.run(data_dirty.values, gen.ExecMode.CP, scale_factor=scale_factor,
                                generate_rand_transformations=False)
    # univar_stats.Stats.get_stats(data_dirty, new_data_dirty)

    print("Original stats:")
    univar_stats.Stats.get_stats(data_clean, data_dirty)
    print("\nScaled stats after random imputation:")
    univar_stats.Stats.get_stats(new_data_clean, new_data_dirty)


def scale_both_and_rand_transform_clean(data_path_clean: str, data_path_dirty: str, is_header: int, scale_factor: float):
    data_clean = pd.read_csv(data_path_clean, header=is_header, sep=',')
    _, new_data_clean = gen.run(data_clean, gen.ExecMode.CP, scale_factor=scale_factor,
                          generate_rand_transformations=True)
    # univar_stats.Stats.get_stats(data_clean, new_data_clean)

    data_dirty = pd.read_csv(data_path_dirty, header=is_header, sep=',')
    _, new_data_dirty = gen.run(data_dirty, gen.ExecMode.CP, scale_factor=scale_factor,
                                generate_rand_transformations=False)
    # univar_stats.Stats.get_stats(data_dirty, new_data_dirty)

    print("Original stats:")
    univar_stats.Stats.get_stats(data_clean, data_dirty)
    print("\nScaled stats after random transformations:")
    univar_stats.Stats.get_stats(new_data_clean, new_data_dirty)


if __name__ == '__main__':
    scale_both_and_rand_transform_clean('datasets/heart.csv', 'datasets/gen_heart.csv', 0, 3)
    # scale_both('datasets/heart.csv', 'datasets/gen_heart.csv', 0, 3)
