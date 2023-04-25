import argparse
import os
import distutils

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
    # Table 1 - all errors
    # meanOrgString = "Outliers original dirty:"
    # meanGenString = "Outliers generated dirty:"

    # meanOrgString = "Typos original dirty:"
    # meanGenString = "Typos generated dirty:"

    # meanOrgString = "MV original dirty:"
    # meanGenString = "MV generated dirty:"

    # meanOrgString = "Replacements original dirty:"
    # meanGenString = "Replacements generated dirty:"

    # meanOrgString = "Unique swaps count - original dirty:"
    # meanGenString = "Unique swaps count - generated dirty:"

    # Table 2
    # meanOrgString = "Distinct estimated typo:"
    # meanGenString = "Unique typos count - original dirty:"

    # meanOrgString = "Distinct estimated mv:"
    # meanGenString = "Unique MV count - generated dirty:"

    # meanOrgString = "Distinct original dirty:"
    # meanOrgString = "Distinct estimated dirty (replicate + errors):"
    # meanGenString = "Distinct generated dirty:"

    meanOrgString = "Mean original dirty:"
    meanGenString = "Mean generated dirty:"


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
    data = [(x,y) for x,y  in zip(sizes, data)]
    return data
    

if __name__ == '__main__':
    args = get_args()
    # args =
    data = parse(args.directory, args.name, args.local)

    print("  ")
    print(args.name)
    data.sort()
    # print(data)
    for x in data:
        print(x[0])
        print("________")
        for dd in x[1]:
            print(sum(dd))
            print("________\n")
