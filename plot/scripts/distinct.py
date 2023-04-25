
import argparse
import ast
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

font = {'family': 'serif',
            'weight': 'normal',
            'size': 12}

matplotlib.rc('font', **font)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True)
    parser.add_argument("-o", "--output", required=True)
    return parser.parse_args()


def parse_dict(line: str):
    return ast.literal_eval(line.strip())


def parse_distinct_frequencies(path: str):
    data = {"clean": {}, "dirty": {}}
    keyword_clean = "Clean distincts"
    keyword_dirty = "Dirty distincts"
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                if keyword_clean in line[:len(keyword_clean)]:
                    data["clean"] = parse_dict(line[len(keyword_clean):])
                if keyword_dirty in line[:len(keyword_dirty)]:
                    data["dirty"] = parse_dict(line[len(keyword_dirty):])
    else:
        print("Did not find file: " + path)
    return data


def make_dict_to_sorted_frequency_list(data: dict):
    l = list(data.values())
    l.sort(reverse=True)
    return l


def calc_yticks(max):
    yticks = []
    s = 1
    while(s <= max):
        yticks.append(int(s))
        s = s * 10
    return yticks


def bar_plot(data, dir, name, color):
    fix, ax = plt.subplots(1, 1,
                           num=None,  figsize=((42) / 6 * 1.0 / 2, 3 * 0.7),
                           dpi=80,
                           facecolor="w",
                           edgecolor="k",)

    ax.bar(range(len(data)), data, 1, color=color)
    ax.margins(x=0)
    ax.set_yscale("log")
    ax.set_yticks(calc_yticks(max(data)))
    ax.set_xticks([len(data)-1])
    ax.set_xlabel(len(data))
    ax.set_ylabel("# Distinct Values")
    ax.set_xlabel("Distinct Data Column")
    ax.yaxis.set_label_coords(-0.13, 0.43)
    ax.xaxis.set_label_coords(0.5, -0.04)

    plt.subplots_adjust(
        left=0.2, right=0.98, top=0.90, bottom=0.05, wspace=0.35, hspace=0.35
    )
    plt.savefig(dir + "/" + name)
    plt.close()


def bar_plot_both(clean, dirty, dir, name):
    fix, ax = plt.subplots(1, 1,
                           num=None,  figsize=((42) / 6 * 1.0 / 2, 3 * 0.7),
                           dpi=80,
                           facecolor="w",
                           edgecolor="k",)

    ax.bar(np.arange(-0.25, len(clean)-0.25, 1), clean, 0.5, color='cornflowerblue', label='Clean')
    ax.bar(np.arange(0.25, len(clean)+0.25, 1), dirty, 0.5, color='orangered', label='Dirty')

    ax.margins(x=0)
    ax.set_yscale("log")
    ax.set_yticks(calc_yticks(max(max(dirty), max(clean))))
    ax.set_xticks([len(dirty)-1])
    ax.set_xlabel(len(data))
    ax.set_ylabel("# Distinct Values")
    ax.set_xlabel("Distinct Data Column")
    ax.yaxis.set_label_coords(-0.15, 0.43)
    ax.xaxis.set_label_coords(0.5, -0.05)
    ax.legend(
        ['Clean', 'Dirty'],
        ncol=2,
        loc="center",
        bbox_to_anchor=(0.5, 1.145),
        # fontsize=7.45,
        #  frameon=False
    )
    plt.subplots_adjust(
        left=0.18, right=0.98, top=0.81, bottom=0.13, wspace=0.35, hspace=0.35
    )

    plt.savefig(dir + "/" + name)
    plt.close()


if __name__ == '__main__':

    args = get_args()
    data = parse_distinct_frequencies(args.file)
    # process
    clean = make_dict_to_sorted_frequency_list(data["clean"])
    dirty = make_dict_to_sorted_frequency_list(data["dirty"])

    # make output dir if not existing
    Path(args.output).mkdir(parents=True, exist_ok=True)

    if(len(clean) > 0):
        bar_plot(dirty, args.output, "clean.pdf", "tab:blue")

    if(len(dirty) > 0):
        bar_plot(dirty, args.output, "dirty.pdf", "tab:red")

    if len(dirty) > 0 and len(clean) > 0:
        bar_plot_both(clean, dirty, args.output, "combined.pdf")

    # plot
