
# dataset='../datasets/UCRArchive_2018/electricity/electricity.csv'
# df_data = pd.read_csv(dataset,header=None,sep='\t|;|:|,| ')
# print(df_data.shape)
# print(df_data.size)
# df_data=denoise(df_data)
# print(df_data.shape)
# print(df_data.size)


import errno
import time
from decimal import Decimal
import statistics as s
from scipy import stats
from tqdm import tqdm
import argparse
import sys
import ntpath
import os
import pandas as pd
import numpy as np
import warnings
from preprocess_wavelets import *
from input_output import *


warnings.filterwarnings("ignore")

def exportResults(name, dic, config):
    df = pd.DataFrame(dic)
    print(dic)

    #
    df = df.rename(index={0:"DTW"})
    print(df)

    download_dir = name  # where you want the file to be downloaded to

    if not os.path.exists(os.path.dirname(download_dir)):
        try:
            os.makedirs(os.path.dirname(download_dir))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    csv = open(download_dir, "a")
    csv.write(config + "\n")
    csv.close()

    df.rename(index={0: "TRISTAN", 1: "y", 2: "z"})
    df.to_csv(download_dir, mode='a', sep='\t', header=True, index=True)

    csv = open(download_dir, "a")
    csv.write("\n\n\n")
    csv.close()


if __name__ == "__main__":

    print("Number of arguments:", len(sys.argv), "arguments.")
    print("Argument List:", str(sys.argv))

    for i in range(len(sys.argv)):
        print(i, sys.argv[i])

    parser = argparse.ArgumentParser(description='Script for running the compression')
    parser.add_argument('--dataset', nargs='?', type=str, help='Dataset path', default='../datasets/UCRArchive_2018/exchangerate/exchangerate.csv')
    # parser.add_argument('--datasetPathDictionary', nargs = '?', type = str, help = 'Dataset path of the dictionary', default = '../datasets/archive_ics/gas-sensor-array-temperature-modulation/20160930_203718-2.csv')
    parser.add_argument('--trick', nargs='?', type=int, help='Length of a tricklet', default=40)
    parser.add_argument('--err', nargs='?', type=float, help='Maximum level of threshold', default=0.1)
    parser.add_argument('--atoms', nargs='?', type=int, help='Number of atoms', default=4)
    # parser.add_argument('--export', nargs = '*', type = str, help = 'Path to file where to export the results', default = 'results.txt')
    parser.add_argument('--additional_arguments', nargs='?', type=str,
                        help='Additional arguments to be passed to the scripts', default='')
    args = parser.parse_args()

    dataset = args.dataset

    # datasetPathDictionary = args.datasetPathDictionary
    trick = args.trick
    err = args.err
    atoms = args.atoms



    TIMESTAMP = time.time()
    CORR_THRESHOLD = 1 - err / 2

    # READING THE DATASETS

    df_data0 = pd.read_csv(dataset,header=None,sep='\t|;|:|,| ')
    df_data0 = df_data0
    # df_data = pd.read_csv(dataset, header=None, sep='\t|;|:|,| ')

    print(df_data0.shape)
    print(df_data0.head())

    # # use wavelet to denoise
    start1 = time.time()
    df_data1 = denoise(df_data0)
    end1 = time.time()

    print(df_data1.shape)


    df_data1 = pd.DataFrame(stats.zscore(df_data1))


    # CREATING TRICKLETS
    # time_series_data = dataframeToTricklets(df_data, trick)         # segment along with quantization
    time_series_data=df_data0





    # # SAVING DATA TO THE DISK
    save_object(
        df_data0, "results/compressed_data/" + str(ntpath.basename(dataset)) + "/originalData.out"
    )

    save_object(
        df_data1, "results/compressed_data/" + str(ntpath.basename(dataset)) + "/DWT_pickle.out"
    )

    dic = {}
    # PRINTING COMPUTATION TIME
    print("Computation DWT time: ", float(round(Decimal(end1 - start1), 2)), "s")

    dic["runtime"] = [
        float(round(Decimal(end1 - start1), 2)),
    ]


    # COMPUTING COMPRESSION RATIOS
    import os

    statinfo = os.stat("results/compressed_data/" + str(ntpath.basename(dataset)) + "/originalData.out")
    statinfo = statinfo.st_size

    statinfo_DWT = os.stat(
        "results/compressed_data/" + str(ntpath.basename(dataset)) + "/DWT_pickle.out"
    )
    statinfo_DWT = statinfo_DWT.st_size

    dic["size_original_(kb)"] = [statinfo / 1024]
    dic["compressed_size_(kb)"] = [statinfo_DWT/1024]


    dic["compression_ratio"] = [
        dic["size_original_(kb)"][0] / (statinfo_DWT / 1024),
    ]

    exportResults(
        "results/"+ str(ntpath.basename(dataset))+ ".txt",
        dic,
        "# config: rmse_error="+ str(err),
    )








