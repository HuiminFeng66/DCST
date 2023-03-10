import errno

from lib.lib import *
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
from lib.preprocess_wavelets import *


warnings.filterwarnings("ignore")

def exportResults(name, dic, config):
    df = pd.DataFrame(dic)
    print(dic)

    # 
    df = df.rename(index={0: "Mine", 1: "CORAD", 2: "TRISTAN"})
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
    parser.add_argument('--dataset', nargs='?', type=str, help='Dataset path', default='datasets/UCRArchive_2018/electricity/electricity_test200.csv')
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

    # dataset = sys.argv[1]
    # datasetPath = sys.argv[2]
    # datasetPathDictionary = sys.argv[3]
    # # NBWEEKS = sys.argv[2]
    # trick = int(sys.argv[4])
    # err = float(sys.argv[5])
    # # trick = NBWEEKS * 7
    # atoms = int(sys.argv[6])

    TIMESTAMP = time.time()
    CORR_THRESHOLD = 1 - err / 2

    # READING THE DATASETS

    df_data = pd.read_csv(dataset,header=None,sep='\t|;|:|,| ')
    df_data = df_data
    # df_data = pd.read_csv(dataset, header=None, sep='\t|;|:|,| ')

    print(df_data.shape)
    print(df_data.head())

    # # use wavelet to denoise
    df_data=denoise(df_data)
    print(df_data.shape)

    # # pre process df_data to reduce memory
    # df_data=reduce_mem_usage(df_data)
    # print('df_data.shape:', df_data.shape)
    # print('df_data.dtypes:', df_data.dtypes)
    # print('df_data.head():' + '\n', df_data.head())

    # df_data_learning = pd.read_csv(datasetPathDictionary, sep='\t|;|:|,| ')
    # df_data_learning = df_data_learning.T

    df_data = pd.DataFrame(stats.zscore(df_data))
    df_data_learning = df_data.iloc[:, 1:3]

    # z-score normalizing the data

    # df_data.round(6)
    # print(df_data.head())
    # df_data.to_csv('yoga_before.txt', header=False, float_format='%.6f', sep='\t|;|:|,| ', index=False)
    # df_data.plot()
    # plt.draw()
    # df_data_learning = pd.DataFrame(stats.zscore(df_data_learning))

    # CREATING TRICKLETS
    time_series_data = dataframeToTricklets(df_data, trick)         # segment along with quantization
    time_series_data_dictionary = dataframeToTricklets(df_data_learning, trick)

    # CORRELATION COMPUTATION FOR EACH SEGMENT( different time series under the same window)
    # without utilizing quantization
    print("Computing correlation1 ... ", end="")
    correlation_matrix1 = []
    for i in tqdm(range(int(df_data.shape[0] / trick))):
        correlation_matrix1.append(
            df_data[i * trick: (i + 1) * trick].corr()
        )
    print("correlation1 computation\ndone!")

    # CORRELATION COMPUTATION FOR EACH SEGMENT( a time series under different window)
    print("Computing correlation2 ... ", end="")
    correlation_matrix2_temp = []   # 208 [50*50]
    correlation_matrix2=[]
    for i in range(int(df_data.shape[1])):
        length = int(int(df_data.shape[0] / trick) * trick)
        x = df_data.iloc[:length, i:i + 1]
        d = np.array(x).reshape(trick, int(length / trick))
        correlation_matrix2_temp.append(
            pd.DataFrame(d).corr()
        )
    for i in tqdm(range(int(df_data.shape[0]/trick))):  # 0~50
        temp=[]
        for j in range(len(correlation_matrix2_temp)):
            temp.append(correlation_matrix2_temp[j].loc[i])
        temp=np.array(temp)
        correlation_matrix2.append(temp)
    for index in range(len(correlation_matrix2)):   # convert list to matrix
        correlation_matrix2[index]=pd.DataFrame(correlation_matrix2[index])
    print("correlation2 computation\ndone!")



    # # using quantization to calculate correlation matrix
    # print("Computing correlation1 ... ", end="")
    # correlation_matrix = []
    # ts_data = np.array(time_series_data)
    # data_ans = []
    # col = 0
    # for row in range(0, ts_data.shape[1]):
    #     temp = []
    #     while col < ts_data.shape[0]:
    #         if len(temp) == 0:
    #             temp = ts_data[col][row]
    #         else:
    #             temp = np.column_stack((temp, ts_data[col][row]))
    #         col += 1
    #     data_ans.append(temp)
    #     col = 0
    #     row += 1
    #
    # data_input = []
    # data_ans_array = np.array(data_ans)
    # for row1 in range(0, data_ans_array.shape[0]):
    #     if len(data_input) == 0:
    #         data_input = data_ans_array[row1]
    #     else:
    #         data_input = np.row_stack((data_input, data_ans_array[row1]))
    #     row1 += 1
    #
    # df_data1 = pd.DataFrame(data_input)
    # for i in tqdm(range(int(df_data1.shape[0] / trick))):
    #     correlation_matrix.append(
    #         df_data1[i * trick: (i + 1) * trick].corr()
    #     )
    # print("correlation1 computation\ndone!")

    # DICTIONARY
    print("Building the dictionary ... ", end="")
    for i in tqdm(range(1, int(len(time_series_data_dictionary)))):
        time_series_data_dictionary[0].extend(time_series_data_dictionary[i])
    print("Learning dictionary")
    Dictionary = learnDictionary(
        time_series_data_dictionary[0], 200, 1, 150, dataset + ".pkl"
    )
    print("done!")

    # COMPRESSING THE DATA THE TRISTAN WAY
    start1 = time.time()
    TRISTAN_atoms_coded_tricklets, errors_TRISTAN = compress_without_correlation(
        time_series_data, Dictionary, atoms, "omp"
    )
    end1 = time.time()

    # COMPRESSING THE DATA CORAD WAY
    start2 = time.time()
    CORAD_atoms_coded_tricklets, corr_coded_tricklets, errors_CORAD = compress_with_correlation1(
        time_series_data,
        correlation_matrix1,
        Dictionary,
        CORR_THRESHOLD,
        atoms,
        "omp",
    )
    end2 = time.time()


    # COMPRESSING THE DATA Mine WAY
    # corr_coded_tricklets1 represents different time sereis under the same window's correlation
    # corr_coded_tricklets2 represents a time series under different windows's correlation
    start3=time.time()
    Mine_atoms_coded_tricklets,corr_coded_tricklets1, corr_coded_tricklets2, errors_Mine=compress_with_correlation2(
        time_series_data,
        correlation_matrix1,
        correlation_matrix2,
        Dictionary,
        CORR_THRESHOLD,
        atoms,
        "omp",
    )
    end3=time.time()

    # # SAVING DATA TO THE DISK
    save_object(
        time_series_data, "results/compressed_data/" + str(ntpath.basename(dataset)) + "/originalData.out"
    )
    save_object(
        TRISTAN_atoms_coded_tricklets,
        "results/compressed_data/" + str(ntpath.basename(dataset)) + "/TRISTAN_pickle.out",
    )
    save_object(
        (CORAD_atoms_coded_tricklets, corr_coded_tricklets),
        "results/compressed_data/" + str(ntpath.basename(dataset)) + "/CORAD_pickle.out",
    )

    save_object(
        (Mine_atoms_coded_tricklets, corr_coded_tricklets1,corr_coded_tricklets2),
        "results/compressed_data/" + str(ntpath.basename(dataset)) + "/Mine_pickle.out",
    )


    dic = {}
    # PRINTING COMPUTATION TIME
    print("Computation time without correlation: ", float(round(Decimal(end1 - start1), 2)), "s")
    print("Computation time with correlation: ", float(round(Decimal(end2 - start2), 2)), "s")
    print("Computation time with auto-inter correlations: ", float(round(Decimal(end3 - start3), 2)), "s")

    # dic['compression_time_without_correltion']= float(round(Decimal(end1 - start1), 2))
    # dic['compression_time_with_correltion']= float(round(Decimal(end2 - start2), 2))
    dic["runtime"] = [
        float(round(Decimal(end3 - start3), 2)),
        float(round(Decimal(end2 - start2), 2)),
        float(round(Decimal(end1 - start1), 2)),
    ]

    # PRINTING ERRORS
    print("Mine error:", "{0:.5}".format(s.mean(errors_Mine)))
    print("CORAD error:", "{0:.5}".format(s.mean(errors_CORAD)))
    print("TRISTAN error:", "{0:.5}".format(s.mean(errors_TRISTAN)))

    # dic['error_CORAD'] = "{0:.5}".format(s.mean(errors_CORAD))
    # dic['error_TRISTAN'] = "{0:.5}".format(s.mean(errors_TRISTAN))
    dic["rmse"] = [
        "{0:.5}".format(s.mean(errors_Mine)),
        "{0:.5}".format(s.mean(errors_CORAD)),
        "{0:.5}".format(s.mean(errors_TRISTAN)),
    ]

    # COMPUTING COMPRESSION RATIOS
    import os

    statinfo_TRISTAN = os.stat(
        "results/compressed_data/" + str(ntpath.basename(dataset)) + "/TRISTAN_pickle.out"
    )
    statinfo_TRISTAN = statinfo_TRISTAN.st_size

    statinfo_CORAD = os.stat(
        "results/compressed_data/" + str(ntpath.basename(dataset)) + "/CORAD_pickle.out"
    )
    statinfo_CORAD = statinfo_CORAD.st_size

    statinfo_Mine = os.stat(
        "results/compressed_data/" + str(ntpath.basename(dataset)) + "/Mine_pickle.out"
    )
    statinfo_Mine = statinfo_Mine.st_size

    statinfo = os.stat("results/compressed_data/" + str(ntpath.basename(dataset)) + "/originalData.out")
    statinfo = statinfo.st_size

    dic["size_original_(kb)"] = [statinfo / 1024, statinfo / 1024, statinfo / 1024]
    dic["compressed_size_(kb)"] = [statinfo_Mine/1024 ,statinfo_CORAD / 1024, statinfo_TRISTAN / 1024]


    dic["compression_ratio"] = [
        dic["size_original_(kb)"][0] / (statinfo_Mine / 1024),
        dic["size_original_(kb)"][0] / (statinfo_CORAD / 1024),
        dic["size_original_(kb)"][0] / (statinfo_TRISTAN / 1024),
    ]

    exportResults(
        "results/"+ str(ntpath.basename(dataset))+ ".txt",
        dic,
        "# config: rmse_error="+ str(err)+ ", atoms="+ str(atoms)+ ", trick="+ str(trick),
    )
