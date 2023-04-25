import random

import scale_modify
from static_utils import get_full_path_str

random.seed(777)


def run_tax_test_distributed():
    dataset = 'tax'
    generated_data = scale_modify.run_distributed(
        clean_path=f'datasets/clean_{dataset}.csv', clean_header=0,
        dirty_path=f'datasets/dirty_{dataset}.csv', dirty_header=0,
        scaling_factor=4,
        original_dist_file=f'err_dist_files/distributed_original_dist_{dataset}.txt',
        gen_dist_file=f'err_dist_files/distributed_gen_dist_{dataset}.txt',
        output_path=f'generated_datasets/gen_{dataset}'
    )

    # print(generated_data.shape)


if __name__ == '__main__':
    run_tax_test_distributed()
