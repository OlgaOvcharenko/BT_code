
import argparse
import ast
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np

font = {'family': 'serif',
            'weight': 'normal',
            'size': 12}

matplotlib.rc('font', **font)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True)
    parser.add_argument("-d", "--datasets", nargs='+', required=True)
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


def parse_errors(path: str, files: list):
    ret = {}
    keywords = ["mv", "outliers", "typos", "swaps", "replacements"]
    for d in files:
        data = {}
        with open(path + d + ".txt") as f:
            for line in f:
                key = line[:21]
                if "Dirty" in key:
                    type = key.split("\t", 1)[0][6:]
                    if type in keywords:
                        data[type] = line.split("\t")[1]
            ret[d] = data
    return ret


def aggregate_errors(data: dict):
    dicts = ["mv", "outliers", "typos", "swaps", "replacements"]
    val = ["swaps"]
    for k, dataset in data.items():
        newDataset = {}
        for kk, v in dataset.items():
            if kk in val:
                newDataset[kk] = int(v.strip())
            else:
                newDataset[kk] = sum(parse_dict(v).values())

        data[k] = newDataset

    return data


def cum_sum_sorted(data: dict, order: list):

    for k, dataset in data.items():
        cum_sum = []
        s = 0
        for o in order:
            s += dataset[o]
            cum_sum.append(s)
        cum_sum.reverse()
        data[k] = cum_sum
    order.reverse()
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


def transpose_across_datasets(data: dict, ordering: list):
    ret = []
    for o in data[ordering[0]]:
        ret.append([])  # allocate one sequence of bars for each error type

    for o in ordering:
        for idx, v in enumerate(data[o]):
            ret[idx].append(v)
    return ret


def cum_sum_to_percent(data:dict):
    # print(data)
    ret = {}
    for k, v in data.items():
        ret[k] = []
        for vv in v:

            ret[k].append(vv / v[0] * 100)
    # print(ret)
    return ret


def plot_stacked_barplot(data, xlabels, legend, name, y_label= "# Errors"):
    fix, ax = plt.subplots(
        1, 1,
        num=None,  figsize=((42) / 6 * 1.8 / 2, 3 * 0.7),
        dpi=80,
        facecolor="w",
        edgecolor="k",)
    colors = ["brown", "tomato", "yellowgreen", "palevioletred" , "cornflowerblue"]
    for idx, err in enumerate(data):
        ax.bar(range(len(err)), err, label=legend[idx], color=colors[idx])
    
    ax.margins(x=0)

    # ax.set_yscale("log")
    # ax.set_yticks(calc_yticks(max(max(dirty), max(clean))))
    ax.set_xticks(range(len(xlabels)))
    # ax.set_xlabel(range(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_ylabel(y_label)
    # ax.set_xlabel("Distinct Data Column")
    ax.set_xlabel("")
    ax.yaxis.set_label_coords(-0.13, 0.43)
    # ax.xaxis.set_label_coords(0.5, -0.14)

    ax.legend(
        ncol=5,
         loc="upper center", 
        bbox_to_anchor=(0.46, 1.25),
        fontsize=9.3,
        #  frameon=False
    )

    # fix.autofmt_xdate(rotation=0)  # fix.autofmt_xdate(rotation=15)
    # dy = 5/72.
    # for idx, label in enumerate(ax.xaxis.get_majorticklabels()):
    #     dx = 0.2  # dx =  6.3/72.
    #     offset = matplotlib.transforms.ScaledTranslation(dx, dy, fix.dpi_scale_trans)
    #     label.set_transform(label.get_transform() + offset)

    plt.subplots_adjust(
        left=0.15, right=0.98, top=0.83, bottom=0.18, wspace=0.35, hspace=0.35
    )
    # plt.grid(True,"major",axis='y', linewidth=0.3, alpha=0.8, ls='--')  # ls='--',
    plt.savefig(name)
    plt.close()


if __name__ == '__main__':

    args = get_args()
    data = parse_errors(args.file, args.datasets)
    data_agg = aggregate_errors(data)
    colum_ordering = ["typos", "mv", "replacements", "swaps", "outliers"]
    # print(data_agg)
    data_cumulative_sums = cum_sum_sorted(data_agg, colum_ordering)

    dataset_ordering = args.datasets
    dataset_ordering.sort()

    

    data_transposed_across_dataset = transpose_across_datasets(
        data_cumulative_sums, dataset_ordering)
    fine_column_names = ["Typo", "MissingValue", "Replacement", "Swap", "Outlier"]
    fine_column_names.reverse()
    plot_stacked_barplot(
        data_transposed_across_dataset,
        dataset_ordering, fine_column_names, args.output+".pdf")

    data_cumulative_percent = cum_sum_to_percent(data_cumulative_sums)
    data_transposed_across_dataset_percent = transpose_across_datasets(
        data_cumulative_percent, dataset_ordering)
    plot_stacked_barplot(
        data_transposed_across_dataset_percent,
        dataset_ordering, fine_column_names, args.output+"_percent.pdf",
        y_label="Percentage of Errors")

    # data = parse_distinct_frequencies(args.file)
    # # process
    # clean = make_dict_to_sorted_frequency_list(data["clean"])
    # dirty = make_dict_to_sorted_frequency_list(data["dirty"])

    # # make output dir if not existing
    # Path(args.output).mkdir(parents=True, exist_ok=True)

    # if(len(clean) > 0):
    #     bar_plot(dirty, args.output, "clean.pdf", "tab:blue")

    # if(len(dirty) > 0):
    #     bar_plot(dirty, args.output, "dirty.pdf", "tab:red")

    # if len(dirty) > 0 and len(clean) > 0:
    #     bar_plot_both(clean, dirty, args.output, "combined.pdf")

    # plot
