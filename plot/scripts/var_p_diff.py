import argparse
import math
import os
import distutils
from fractions import Fraction

import matplotlib
import numpy as np
from matplotlib import pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", required=True)
    parser.add_argument("-n", "--name", required=True)
    parser.add_argument("-l", "--local", required=True, choices=('True','False'))
    parser.add_argument("-o", "--output", required=True)

    args = parser.parse_args()
    args.local = args.local == "True"
    return args


def parse_single_file(dict:str, path: str):
    meanOrgString = "Var original dirty:"
    meanGenString = "Var generated dirty:"


    meanOrgLen = len(meanOrgString)
    meanGenLen = len(meanGenString)
    meanOrg = []
    meanGen = []
    with open(dict + "/" + path) as f:
        for line in f:
            if meanOrgString in line:
                meanOrg.append(float(line[meanOrgLen:].strip()))

            if meanGenString in line:
                meanGen.append(float(line[meanGenLen:].strip()))

    return (meanOrg, meanGen)
    pass


def parse(directory: str, name: str,  local: bool):
    print(directory)
    print(local)
    if local:
        l = "local"
    else:
        l = "distributed"

    # files = [f for f in os.listdir(directory) if (name in f and l in f)]
    files = [f for f in os.listdir(directory) if (name in f)]
    print(files)

    sizes = [int(f.split("_")[1]) for f in files]
    data = [parse_single_file(directory, f) for f in files]
    data = [(x, y) for x, y in zip(sizes, data)]
    return data


def bar_plot_both(clean, dirty, dir, name):
    fix, ax = plt.subplots(1, 1,
                           num=None,  figsize=((42) / 6 * 1.0 / 2, 3 * 0.7),
                           dpi=80,
                           facecolor="w",
                           edgecolor="k",)

    ax.bar(np.arange(-0.25, len(clean)-0.25, 1), clean, 0.5, color='tab:blue')
    ax.bar(np.arange(0.25, len(clean)+0.25, 1), dirty, 0.5, color='tab:red')

    ax.margins(x=0)
    ax.set_yscale("log")
    # ax.set_yticks(calc_yticks(max(max(dirty), max(clean))))
    ax.set_xticks([len(dirty)-1])
    ax.set_xlabel(len(data))
    ax.set_ylabel("# Distinct Values")
    ax.set_xlabel("Distinct Data Column")
    ax.yaxis.set_label_coords(-0.13, 0.43)
    ax.xaxis.set_label_coords(0.5, -0.04)

    plt.subplots_adjust(
        left=0.15, right=0.98, top=0.96, bottom=0.1, wspace=0.35, hspace=0.35
    )
    plt.savefig(dir + "/" + name)
    plt.close()


def calc_yticks(max):
    yticks = []
    s = 0.0

    to_add = max / 3
    print(to_add)
    while s <= max:
        yticks.append(round(s, 2))
        s = s + to_add

    print(yticks)
    return yticks


def plot_scatter_plots(scales_x, diff_y, path):
    fix, ax = plt.subplots(1, 1,
                           num=None, figsize=((42) / 6 * 1.0 / 2, 3 * 0.7),
                           dpi=80,
                           facecolor="w",
                           edgecolor="k", )

    # ax.margins(x=0, y=0)

    ticks_y = calc_yticks(max(diff_y)) if max(diff_y) != 0.0 else [0.0]
    ax.set_yticks(ticks_y)
    ax.set_yticklabels(ticks_y)

    ax.set_xlabel("Scaling Factors")
    ax.set_ylabel("% Var Difference")
    ax.yaxis.set_label_coords(-0.24, 0.43)
    plt.subplots_adjust(
        left=0.23, right=0.98, top=0.96, bottom=0.2, wspace=0.35, hspace=0.35
    )
    ax.set_xscale("log")
    ax.set_xticks(scales_x)
    ax.set_xticklabels(scales_x)
    plt.plot(scales_x, diff_y, "-ok", "")
    plt.savefig(path)
    # plt.show()
    plt.close()


def plot_all_scatter_plots(names, scales_x, diff_y, ticks, path):
    font = {'family': 'serif',
            'weight': 'normal',
            'size': 12}

    matplotlib.rc('font', **font)

    fix, ax = plt.subplots(1, 1,
                           num=None, figsize=((42) / 4, 3 * 1.5 / 2),
                           dpi=30,
                           facecolor="w",
                           edgecolor="k", )

    ax.margins(x=0.01)

    # ticks_y = calc_yticks(max(diff_y)) if max(diff_y) != 0.0 else [0.0]
    # ax.set_yticks(diff_y[0])
    # ax.set_yticklabels(diff_y[0])

    ax.set_xlabel("Scaling Factors")
    ax.set_ylabel("% Variance Difference")
    # ax.yaxis.set_label_coords(-0.24, 0.43)
    plt.subplots_adjust(
        left=0.07, right=0.99, top=0.80, bottom=0.22, wspace=0.35, hspace=0.35
    )
    ax.set_xscale("log")
    ax.set_xticks(scales_x)
    ax.set_xticklabels(scales_x)

    ax.grid(axis="y")

    ax.set_yscale("log")
    y_ticks = [math.pow(10, -4), math.pow(10, -3), math.pow(10, -2), math.pow(10, -1),
               math.pow(10, 0), math.pow(10, 1), math.pow(10, 2)]
    ax.set_yticks(y_ticks)

    colors = ["brown", "tomato", "yellowgreen", "crimson", "cornflowerblue", "darkgreen"]
    markers = ["*", "o", "v", "x", "X", "s"]
    for d, y, c, m in zip(names, diff_y, colors, markers):
        plt.plot(scales_x, y, "", label=d, color=c, marker=m)

    plt.plot(scales_x, [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], label='Allowed difference', color='black', linestyle="--")

    plt.legend(ncol=7, loc='center', bbox_to_anchor=(0.48, 1.2))
    plt.savefig(path)
    # plt.show()
    plt.close()

# def plot_all_scatter_plots(names, scales_x, diff_y, ticks, path):
#     fix, ax = plt.subplots(1, 1,
#                            num=None, figsize=((42) / 4 * 1.0 / 2, 3 * 0.7),
#                            dpi=80,
#                            facecolor="w",
#                            edgecolor="k", )
#
#     # ax.margins(x=0, y=0)
#
#     # ticks_y = calc_yticks(max(diff_y)) if max(diff_y) != 0.0 else [0.0]
#     # ax.set_yticks(diff_y[0])
#     # ax.set_yticklabels(diff_y[0])
#
#     ax.set_xlabel("Scaling Factors")
#     ax.set_ylabel("% Var Difference")
#     ax.yaxis.set_label_coords(-0.24, 0.43)
#     plt.subplots_adjust(
#         left=0.23, right=0.98, top=0.96, bottom=0.2, wspace=0.35, hspace=0.35
#     )
#     ax.set_xscale("log")
#     ax.set_xticks(scales_x)
#     ax.set_xticklabels(scales_x)
#
#     ax.set_yscale("log")
#     # ax.set_yticks(diff_y[0])
#     # ax.set_yticklabels(diff_y[0])
#
#     colors = ["brown", "tomato", "yellowgreen", "crimson", "cornflowerblue", "darkgreen"]
#     for d, y, c in zip(names, diff_y, colors):
#         plt.plot(scales_x, y, "", label=d, color=c)
#
#     plt.plot(scales_x, [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], label='Allowed diff', color='black', linestyle="--")
#
#     plt.legend(ncol=3)
#     plt.savefig(path)
#     # plt.show()
#     plt.close()


if __name__ == '__main__':
    args = get_args()
    # args =
    data = parse(args.directory, args.name, args.local)

    scales, abs_diff = [], []

    print("  ")
    print(args.name)
    data.sort()
    # print(data)
    for x in data:
        print(x[0])
        print("________")

        scales.append(x[0])

        print(sum(x[1][0]))
        print(sum(x[1][1]))
        val = abs(float(sum(x[1][1]) - sum(x[1][0])) / sum(x[1][1])) * 100 if sum(x[1][1]) != 0 else 0.0
        abs_diff.append(val)
        print(val)
    metric = "var"
    dataset_name = "tax"
    names = ["tax", "rayyan", "movies", "hospital", "flights", "beers"]
    values = [
        [1.213645358338122, 1.6860361016292176, 1.4750664137299516, 27.618431732165067,
         50.00757596967157, 59.44088802260444, 31.122492504696663, 24.1782],

        [0.050050593175407554, 0.07508107496411418, 0.08761274332721403, 0.09386325502434445,
         0.09699684014224662, 0.09856037830521042, 0.0993313428538169, 0.09973124502121967],

        [99.995184717655, 99.99578897940202, 99.99557174271648, 99.99532793947043,
         99.99535003272659, 99.99536502050319, 99.99527951329566, 99.99529899382804],

        [3.1147124055352546, 1.5030195323282398, 1.610664451339219, 1.7957006698565285,
         2.651718288330259, 1.6309188436303956, 2.3445694591769186, 1.8765527420278514],

        [0.021052631578943277, 0.03157894736841836, 0.036842105263162374, 0.03947368421052415,
         0.040789473684216634, 0.04144736842105774, 0.041776315789471954, 0.04194078947368825],

        [0.040714128550783595, 0.04863815244229819, 0.05396478193236943, 0.057466298762901324,
         0.05861810734507447, 0.0594699503125246, 0.059565027845064046, 0.059736821238720594]
    ]

    datasets = [[0.0, 33.5, 67.01], [0.0, 0.05, 0.11, 0.16], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    print(abs_diff)

    plot_all_scatter_plots(names=names, scales_x=scales, diff_y=values, ticks=datasets, path=f'plot/{metric}_diff.pdf')
