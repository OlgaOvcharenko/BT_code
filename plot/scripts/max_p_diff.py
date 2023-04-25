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
    meanOrgString = "Max original dirty:"
    meanGenString = "Max generated dirty:"


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
    ax.set_ylabel("% Min Difference")
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
    ax.set_ylabel("% Max Difference")
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
#     ax.set_ylabel("% Min Difference")
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
#     plt.legend(ncol=2)
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
    metric = "max"
    dataset_name = "tax"
    names = ["tax", "rayyan", "movies", "hospital", "flights", "beers"]
    values = [
        [0.6390732552489693, 0.8514706976593168, 0.6025270693252691, 24.602454472029027,
         49.65662466017235, 49.70784814513716, 49.708701441874815, 49.708701441874815],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [99.76860543523875, 99.80268817726353, 99.80268817726353, 99.80268817726353, 99.80268817726353,
         99.80268817726353, 99.80268817726353, 99.80268817726353],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.00013799936795963146, 0.00013799936795963146, 0.00013799936795963146, 0.00013799936795963146,
         0.00013799936795963146, 0.00013799936795963146, 0.00013799936795963146, 0.00013799936795963146]]
    datasets = [[0.0, 33.5, 67.01], [0.0, 0.05, 0.11, 0.16], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    print(abs_diff)

    plot_all_scatter_plots(names=names, scales_x=scales, diff_y=values, ticks=datasets, path=f'plot/{metric}_diff.pdf')
