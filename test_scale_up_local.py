import argparse
import random

import scale_modify

random.seed(777)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--clean", required=True)
    parser.add_argument("-d", "--dirty", required=True)
    parser.add_argument("-sf", "--scalingFactor", required=True)
    parser.add_argument("-edc", "--errDistC", required=True)
    parser.add_argument("-edd", "--errDistD", required=True)
    parser.add_argument("-o", "--output", required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    generated_data = scale_modify.run_default(
        clean_path=args.clean, clean_header=0,
        dirty_path=args.dirty, dirty_header=0,
        scaling_factor=float(args.scalingFactor),
        original_dist_file=args.errDistC,
        gen_dist_file=args.errDistD,
        output_path=args.output)
    print(generated_data.shape)
    print("+++++++++++++++++++++++++++++++")
