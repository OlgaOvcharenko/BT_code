import random

import scale_modify

random.seed(777)


def run_tax_test_local():
    generated_data = scale_modify.run_default(
        clean_path='datasets/clean_hospital.csv', clean_header=0,
        dirty_path='datasets/dirty_hospital.csv', dirty_header=0,
        scaling_factor=2.0,
        original_dist_file='err_dist_files/local_original_dist_hospital.txt',
        gen_dist_file='err_dist_files/local_gen_dist_hospital.txt',
        output_path='generated_datasets/generated_hospital.csv'
    )
    print(generated_data.shape)


if __name__ == '__main__':
    run_tax_test_local()
