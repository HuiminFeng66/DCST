import matplotlib.pyplot as plt
from datetime import timedelta
import sys
from luminol.anomaly_detector import AnomalyDetector
from luminol.correlator import Correlator
from lib.dictionary_learning import *
from lib.quantization.demo_MaxLloydQuantizer_scalar_quantization import *


def getTrickletsTS(time_series, nbTS, nbweeks):
    # First  timestamp
    s1 = time_series.iloc[:, 0][0]

    # Segment length
    slength = timedelta(weeks=nbweeks)

    # Data segmentation
    i = 0
    ts = [[] for i in range(nbTS)]

    # Read two time series
    while ((s1 + i * slength < time_series.iloc[:, 0][-1])):
        d = time_series[time_series['date'] >= (s1 + i * slength)]
        d = d[d['date'] < (s1 + (i + 1) * slength)]
        # print(d)
        # print(d.iloc[:, 1])
        # print(d.iloc[:, 2])
        for k in range(nbTS):
            if (len(d.iloc[:, k + 1].tolist()) == nbweeks * 7):
                ts[k].append(d.iloc[:, k + 1].tolist())
        i += 1

    # print('ts1')
    # print(ts1)
    # print('ts2')
    # print(ts2)
    return ts


# def getTricklets(data, length, nbTS):
#     # First  timestamp
#     s1 = data.iloc[:, 0][0]
#
#     # Segment length
#     slength = timedelta(weeks=nbweeks)
#
#     # Data segmentation
#     i = 0
#     ts = [[] for i in range(nbTS)]
#
#     # Read two time series
#     while ((s1 + i * slength < time_series.iloc[:, 0][-1])):
#         d = time_series[time_series['date'] >= (s1 + i * slength)]
#         d = d[d['date'] < (s1 + (i + 1) * slength)]
#         # print(d)
#         # print(d.iloc[:, 1])
#         # print(d.iloc[:, 2])
#         for k in range(nbTS):
#             if (len(d.iloc[:, k + 1].tolist()) == nbweeks * 7):
#                 ts[k].append(d.iloc[:, k + 1].tolist())
#         i += 1
#
#     # print('ts1')
#     # print(ts1)
#     # print('ts2')
#     # print(ts2)
#     return ts


# def getTimeStamp(id_tricklet):
#     return id_tricklet * nbweeks * 7
#

def plotData(data):
    plt.plot(data)
    plt.show(block=True)


def plotManyData(data, x, xaxis, yaxiy):
    if xaxis > 1 and yaxiy > 1:
        fig, ax = plt.subplots(nrows=xaxis, ncols=yaxiy)
        i = 0
        for row in ax:
            for col in row:
                col.plot(x, data[i])
                i += 1
    else:
        plt.plot(data[0])

    plt.show(block=True)


def trickletIsIn(timeStamp, corr, length):
    return [item for item in corr if item[0] <= timeStamp and timeStamp + length <= item[1]]


def runSparseCoder(Dictionary, test_data, nonzero_coefs, transform_algorithm):
    from sklearn.decomposition import SparseCoder

    coder = SparseCoder(dictionary=Dictionary, transform_n_nonzero_coefs=nonzero_coefs,
                        transform_alpha=None, transform_algorithm=transform_algorithm)

    print('test_data')
    print(test_data)
    result = coder.transform(test_data)

    tricklets = []
    # tricklets = np.array([np.array([[i,e] for i, e in enumerate(result[t]) if e != 0 for t in range(result.shape[0])])])

    for t in range(result.shape[0]):
        x = []
        for i, e in enumerate(result[t]):
            if e != 0:
                x.append([i, e])

        # print(type(x))
        tricklets.append(x)
        # tricklets= np.append(tricklets, np.array([[i, e] for i, e in enumerate(result[t]) if e != 0]))

    # print(tricklets)
    # print("result size: " + str( *jnu9n *juuricklets.shape))
    # print("result")
    tricklets = np.array([np.array(xi) for xi in tricklets])
    # np.set_printoptions(threshold=np.inf)
    # print(tricklets)
    # print(result)
    # print(result.shape)
    return tricklets


def print_dictionary(dict):
    for key, val in dict.items():
        print(key, "=>", val)

#
# def runSparseCoder_new(Dictionary, test_data, nonzero_coefs, transform_algorithm, corr):
#     from sklearn.decomposition import SparseCoder
#
#     coder = SparseCoder(dictionary=Dictionary, transform_n_nonzero_coefs=nonzero_coefs,
#                         transform_alpha=None, transform_algorithm=transform_algorithm)
#
#     # test_data: list of lists
#     print('test_data000')
#     print(len(test_data))
#     print(type(test_data))
#     print(test_data)
#     code = coder.transform(test_data)
#     # code = coder.transform2Tricklets(test_data)
#     print(code.shape)
#     atoms_coded_tricklets = {}
#     corr_coded_tricklets = {}
#     # tricklets = np.array([np.array([[i,e] for i, e in enumerate(result[t]) if e != 0 for t in range(result.shape[0])])])
#     # print('result:')
#     # print(result)
#
#     for t in range(code.shape[0]):
#         if not trickletIsIn(getTimeStamp(t), corr, 7 * nbweeks):
#             x = []
#             for i, e in enumerate(code[t]):
#                 if e != 0:
#                     x.append([i, e])
#             # print(type(x))
#             atoms_coded_tricklets[t] = x
#         else:
#             corr_coded_tricklets[t] = 1
#
#     # atoms_coded_tricklets = np.array([np.array(xi) for xi in atoms_coded_tricklets])
#
#     # print()
#     # print(atoms_coded_tricklets)
#
#     return atoms_coded_tricklets, corr_coded_tricklets


def reconstructData(sparseData, Dictionary):
    # sparseData [n, m] : n = tricklets number; m: nb atoms
    # Dictionary [n, m] : n = tricklet length; m: nb atoms
    # print(result.shape)

    out = []
    # print(sparseData.shape)
    # print(Dictionary.T.shape)
    for t in range(sparseData.shape[0]):
        sum = np.zeros(Dictionary.T.shape[0])

        for i, e in sparseData[t]:
            # print(Dictionary.T[:,int(i)])
            # print(e)
            # print('\n')
            sum += Dictionary.T[:, int(i)] * e

        out.append(sum)
        # out.append(np.sum(Dictionary.T * sparseData[n], axis=1))

    # print(len(out))
    return out


def reconstructDataMulti_without_correlation(sparseData, Dictionary):
    # sparseData [n, m] : n = tricklets number; m: nb atoms
    # Dictionary [n, m] : n = tricklet length; m: nb atoms
    # print(result.shape)
    # print(sparseData.shape)
    # print(Dictionary.T.shape)
    result = []
    # result = [[] for i in range(len(sparseData))]
    # print(sparseData)
    for index in range(len(sparseData)):
        out = []
        # print(sparseData[index])
        # print()
        for t in range(sparseData[index].shape[0]):
            # print(t)
            sum = np.zeros(Dictionary.T.shape[0])

            for i, e in sparseData[index][t]:
                # print(Dictionary.T[:,int(i)])
                # print(e)
                # print('\n')
                sum += Dictionary.T[:, int(i)] * e

            out.append(sum.tolist())
            # print(out)
        # print(out)
        result.append(out)
        # print(result)

        # out.append(np.sum(Dictionary.T * sparseData[n], axis=1))

    # print(len(out))
    # print(len(result[0]))
    return result


def reconstructDataMulti_with_correlation1(atoms_coded_tricklets, corr_coded_tricklets, Dictionary, ts):
    result = {}
    # start with reconstructing the atoms stored tricklets
    # for each time series
    for k, v in atoms_coded_tricklets.items():
        out = {}
        for w in sorted(v.keys()):
            sum = np.zeros(Dictionary.T.shape[0])
            for i, e in v[w]:
                sum += Dictionary.T[:, int(i)] * e
            out[w] = sum.tolist()
        result[k] = out
    # for each TS stored using correlation
    for k in corr_coded_tricklets.keys():
        # for each window and shift value
        for w in corr_coded_tricklets[k].keys():
            i_m = corr_coded_tricklets[k][w]
            if k not in result.keys():
                result[k] = {}
            result[k][w] = [x  for x in result[i_m][w]]
    resultList = []
    for i in range(len(result.values())):
        resultList.append([result[i][j] for j in sorted(result[i].keys())])
    return resultList


def reconstructDataMulti_with_correlation2(atoms_coded_tricklets,corr_coded_tricklets1,corr_coded_tricklets2,Dictionary,ts):
    result = {}
    # start with reconstructing the atoms stored tricklets
    # for each time series
    for k, v in atoms_coded_tricklets.items():
        out = {}
        for w in sorted(v.keys()):
            sum = np.zeros(Dictionary.T.shape[0])
            for i, e in v[w]:
                sum += Dictionary.T[:, int(i)] * e
            out[w] = sum.tolist()
        result[k] = out
    # for each TS stored using correlation
    for k_1 in corr_coded_tricklets1.keys():
        # for each window and shift value
        for w_1 in corr_coded_tricklets1[k_1].keys():
            i_m_1 = corr_coded_tricklets1[k_1][w_1]
            if k_1 not in result.keys():
                result[k_1] = {}
            result[k_1][w_1] = [x for x in result[i_m_1][w_1]]

    # for each TS stored using correlation
    for k_2 in corr_coded_tricklets2.keys():
        # for each window and shift value
        for w_2 in corr_coded_tricklets2[k_2].keys():
            i_m_2 = corr_coded_tricklets2[k_2][w_2]
            if k_2 not in result.keys():
                result[k_2] = {}
            result[k_2][w_2] = [x for x in result[i_m_2][w_2]]

    resultList = []
    for i in range(len(result.values())):
        resultList.append([result[i][j] for j in sorted(result[i].keys())])
    return resultList



# def find_corr_list(result, corr_coded_tricklets, ts, window, shift):
def find_corr_list(result, corr_coded_tricklets, ts, window):
    try:
        return result[ts][window]
    except:
        print(ts, window)
        raise


def pause():
    input("Press Enter to continue...")


def reconstructData_new(atoms_coded_tricklets, corr_coded_tricklets, otherTS, Dictionary):
    # sparseData [n, m] : n = tricklets number; m: nb atoms
    # Dictionary [n, m] : n = tricklet length; m: nb atoms
    # print(result.shape)

    out = []
    # print(Dictionary.T.shape)

    for t in corr_coded_tricklets:
        atoms_coded_tricklets[t] = otherTS[t] * corr_coded_tricklets[t]
        # atoms_coded_tricklets[t] = otherTS[t] * 0

    for t in sorted(atoms_coded_tricklets.keys()):
        # print(t)
        sum = np.zeros(Dictionary.T.shape[0])

        for i, e in atoms_coded_tricklets[t]:
            # print(Dictionary.T[:,int(i)])
            # print(e)
            # print('\n')
            sum += Dictionary.T[:, int(i)] * e

        out.append(sum)
    # out.append(np.sum(Dictionary.T * sparseData[n], axis=1))

    # for t in range(otherTS.shape[0]):
    #     if t in atoms_coded_tricklets:
    #         sum = np.zeros(Dictionary.T.shape[0])
    #         print(atoms_coded_tricklets[t])
    #         for i, e in atoms_coded_tricklets[t]:
    #             # print(Dictionary.T[:,int(i)])
    #             # print(e)
    #             # print('\n')
    #             sum += Dictionary.T[:, int(i)] * e
    #
    #         out.append(sum)
    #         # out.append(np.sum(Dictionary.T * sparseData[n], axis=1))
    #     else:
    #         out.append(corr_coded_tricklets[t])

    # print(len(out))
    return out


def localCorrelation(ts1, ts2, score_threshold, correlation_threshold, minLength):
    my_detector = AnomalyDetector(ts1, score_threshold=score_threshold)
    score = my_detector.get_all_scores()
    anomalies = my_detector.get_anomalies()

    list = []

    for a in anomalies:
        time_period = a.get_time_window()
        # print(time_period)
        if int(time_period[1]) - int(time_period[0]) >= minLength:
            # print(type(time_period))
            my_correlator = Correlator(ts1, ts2, time_period)
            if my_correlator.is_correlated(threshold=correlation_threshold):
                # print("ts2 correlate with ts1 at time period (%d, %d)" % time_period)
                list.append(time_period)
    return list


# def test_dictionary_building_old(ts1, ts2):
#     # Reading time series
#     # ts1, ts2 = read_time_series('SURF_CLI_CHN_MUL_DAY-TEM_50468-1960-2012-short.txt')
#
#     # Get sample 100 tricklets to reconstruct
#     test_data = np.array(ts2[20:120])
#
#     # Build the dictionary
#     print("Building the dictionary ... ", end='')
#     D = learnDictionary(ts1, 36, 1, 100)
#     print("done!")
#     # print(len(D))
#
#     # Transforming test data into sparse respresentation using the omp algorithm
#     print("Transforming test data into sparse respresentation ... ", end='')
#     sparseData = runSparseCoder(D, test_data, nbAtoms, "omp")
#     # np.set_printoptions(threshold=np.inf)
#     # print(sparseData)
#
#     sparseDataX = pd.DataFrame.from_records(sparseData)
#     # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     #     print(sparseDataX)
#     #     print(len(sparseDataX))
#
#     print("done!")
#
#     print("Reconstructing data...", end="")
#     out = reconstructData(sparseData, D)
#     print("done!")
#
#     print("Plotting.. ", end="")
#     plotManyData(test_data, range(len(D[0])), 2, 2)
#     plotManyData(D, range(len(D[0])), 6, 6)
#     plotManyData(out, range(len(D[0])), 2, 2)
#
#     plt.show(block=True)
#     print("done!")
#
#     print("Error's norm: ", end="")
#     print(np.linalg.norm(np.array(test_data) - np.array(out)))


def dict_to_array(dict):
    dictlist = []
    for key, value in dict.items():
        temp = [key, value]
        dictlist.append(temp)
    return dictlist


# def test_dictionary_building(ts1, ts2, corr):
#     # Reading time series
#     # ts1, ts2 = read_time_series('SURF_CLI_CHN_MUL_DAY-TEM_50468-1960-2012-short.txt')
#
#     # Build the dictionary
#     print("Building the dictionary ... ", end='')
#     D = learnDictionary(ts2, 400, 1, 100)
#     print("done!")
#     # print(len(D))
#
#     # Transforming test data into sparse respresentation using the omp algorithm
#     print("Transforming test data into sparse respresentation ... ", end='')
#     atoms_coded_tricklets1, corr_coded_tricklets1 = runSparseCoder_new(D, ts1, nbAtoms, "omp", corr)
#     # atoms_coded_tricklets1 = dict_to_array(atoms_coded_tricklets1)
#     # corr_coded_tricklets1 = dict_to_array(corr_coded_tricklets1)
#
#     atoms_coded_tricklets2 = runSparseCoder(D, ts1, nbAtoms, "omp")
#
#     # print(runSparseCoder(D, ts1, nbAtoms, "omp"))
#     # print(atoms_coded_tricklets1)
#     # print(corr_coded_tricklets1)
#     old_size = sys.getsizeof(atoms_coded_tricklets2)
#     new_size = sys.getsizeof(atoms_coded_tricklets1) + sys.getsizeof(corr_coded_tricklets1)
#     print('old size: ')
#     print(old_size)
#     print('new size: ')
#     print(new_size)
#
#     print('Compression rate:')
#     print("{0:.0%}".format(1. - float(new_size) / float(old_size)))
#     # # np.set_printoptions(threshold=np.inf)
#     # # print(sparseData)
#
#     # sparseDataX = pd.DataFrame.from_records(sparseData)
#     # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     #     print(sparseDataX)
#     #     print(len(sparseDataX))
#
#     print("done!")
#
#     print("Reconstructing data...", end="")
#
#     reconstruct1 = reconstructData_new(atoms_coded_tricklets1, corr_coded_tricklets1, atoms_coded_tricklets2, D)
#     reconstruct2 = reconstructData(atoms_coded_tricklets2, D)
#     # print(np.array(reconstruct1))
#     # print(np.array(reconstruct2))
#
#     # print(np.array(normalized(reconstruct1)).shape)
#     print("reconstruct1!")
#
#     # print(np.array(normalized(reconstruct2)).shape)
#     print("done!")
#
#     #
#     # # print("Plotting.. ", end="")
#     # # plotManyData(test_data, range(len(D[0])), 2, 2)
#     # # plotManyData(D, range(len(D[0])), 6, 6)
#     # # plotManyData(out, range(len(D[0])), 2, 2)
#     # #
#     # # plt.show(block=True)
#     # # print("done!")
#     # #
#
#     # print(np.array(normalized(ts1)))
#     # print(np.array(normalized(reconstruct1)))
#
#     # print(np.array(reconstruct1) - np.array(reconstruct2))
#
#     print("Error's norm of the correlation-aware method: ", end="")
#
#     # error1 = np.linalg.norm(np.array(normalized(ts1)) - np.array(normalized(reconstruct1)))
#     error1 = calculate_RMSE(ts1, reconstruct1)
#     # # error1 = ((np.array(normalized(ts1) - np.array(normalized(reconstruct1))) ** 2).mean(axis=None))
#     print('%.2E' % Decimal(error1))
#
#     print("Error's norm of the regular method: ", end="")
#     error2 = calculate_RMSE(ts1, reconstruct2)
#
#     # # error2 = ((np.array(normalized(ts2) - np.array(normalized(reconstruct1))) ** 2).mean(axis=None))
#     # print('%.2E' % Decimal(error2))
#
#     # error2 = np.linalg.norm(np.array(normalized(ts1)) - np.array(normalized(reconstruct2)))
#     print('%.2E' % Decimal(error2))
#
#     print('Lost precision: ' + str(error2 - error1))





def sparse_code_without_correlation(ts, Dictionary, nonzero_coefs, transform_algorithm):
    from sklearn.decomposition import SparseCoder

    coder = SparseCoder(dictionary=Dictionary, transform_n_nonzero_coefs=nonzero_coefs
                        , transform_algorithm=transform_algorithm)

    # For each time series, for each tricklet, transform the tricklet and store it
    result = []
    for t in ts:
        result.append(coder.transform(t))

    # transformation of result to [id_A, coef_A]
    tricklets = []
    for index in range(len(result)):
        temp = []
        for t in range(result[index].shape[0]):
            x = []
            for i, e in enumerate(result[index][t]):
                if e != 0:
                    x.append([i, e])
            temp.append(x)
        tricklets.append(temp)

    for index in range(len(result)):
        tricklets[index] = np.array([
            np.array(xi) for xi in tricklets[index]
        ])

    return tricklets


def normalize_df(df):
    from scipy.signal import detrend

    # x = df.values  # returns a numpy array
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_scaled = min_max_scaler.fit_transform(x)
    # df = pd.DataFrame(x_scaled)
    return detrend(df)

def mse(x, y):
    return ((np.array(x) - np.array(y)) ** 2).mean(axis=None)


def sparse_code_with_correlation1(ts, correlation_matrix1, Dictionary, nonzero_coefs, transform_algorithm, threshold):
    """
    :type correlation_matrix1: object
    """
    from sklearn.decomposition import SparseCoder
    coder = SparseCoder(dictionary=Dictionary, transform_n_nonzero_coefs=nonzero_coefs,
                        transform_alpha=None, transform_algorithm=transform_algorithm)
    # For each time series, for each tricklet, transform the tricklet and store it
    result = []
    for t in ts:
        result.append(coder.transform(t))
    # tricklets sparsely coded
    tricklets = []
    # transform sparse matrix into sparse arrays
    for index in range(len(result)):
        temp = []
        for t in range(result[index].shape[0]):
            x = []
            for i, e in enumerate(result[index][t]):
                if e != 0:
                    x.append([i, e])
            temp.append(x)
        tricklets.append(temp)
    for index in range(len(result)):
        tricklets[index] = np.array([np.array(xi) for xi in tricklets[index]])

    atoms_coded_tricklets = {}
    corr_coded_tricklets = {}

    # for each time window
    for w in range(result[0].shape[0]):
        # create dictionary to keep indices
        A = correlation_matrix1[w].values.tolist()
        B = {i: A[i] for i in range(len(A))}
        # sort lines in a decent order by the sum of their elements
        C = dict(sorted(B.items(), key=lambda i: sum(i[1]), reverse=True))

        i_stored = []

        # for each time series
        for k, X in C.items():
            # Find the index maximizing the correlation
            # m = list of indices of corr TS candidates AND already stored normally and different than itself
            m = {i: v for i, v in enumerate(X)
                 if (i in i_stored and v >= threshold and k != i)
                 }
            m = dict(sorted(m.items(), key=lambda i: i[1], reverse=True))

            try:
                i_m = list(m.keys())[0]
            except:
                i_m = None
            if i_m is not None:  # store corr
                x = ts[i_m][w]
                y = ts[k][w]
                if k not in corr_coded_tricklets.keys():
                    corr_coded_tricklets[k] = {}
                corr_coded_tricklets[k][w] = i_m

                # z = [v + shift_mean(x, y) for v in x]
                z = [v for v in x]
            else:  # store sparse
                if k not in atoms_coded_tricklets:
                    atoms_coded_tricklets[k] = {}
                atoms_coded_tricklets[k][w] = tricklets[k][w]
                # add k to the list of elements stored in sparse way
                i_stored.append(k)
    return atoms_coded_tricklets, corr_coded_tricklets



def sparse_code_with_correlation2(ts, correlation_matrix1,correlation_matrix2, Dictionary, nonzero_coefs, transform_algorithm, threshold):
    """
    :type correlation_matrix1: object
    """
    from sklearn.decomposition import SparseCoder
    coder = SparseCoder(dictionary=Dictionary, transform_n_nonzero_coefs=nonzero_coefs,
                        transform_alpha=None, transform_algorithm=transform_algorithm)
    # For each time series, for each tricklet, transform the tricklet and store it
    result = []
    for t in ts:
        result.append(coder.transform(t))
    # tricklets sparsely coded
    tricklets = []
    # transform sparse matrix into sparse arrays
    for index in range(len(result)):
        temp = []
        for t in range(result[index].shape[0]):
            x = []
            for i, e in enumerate(result[index][t]):
                if e != 0:
                    x.append([i, e])
            temp.append(x)
        tricklets.append(temp)
    for index in range(len(result)):
        tricklets[index] = np.array([np.array(xi) for xi in tricklets[index]])

    atoms_coded_tricklets = {}
    corr_coded_tricklets1 = {}
    corr_coded_tricklets2 = {}    # need to modify

    # for each time window
    for w in range(result[0].shape[0]):
        # create dictionary to keep indices
        A = correlation_matrix1[w].values.tolist()
        B = {i: A[i] for i in range(len(A))}
        # sort lines in a decent order by the sum of their elements
        C = dict(sorted(B.items(), key=lambda i: sum(i[1]), reverse=True))

        i_stored_1 = []
        # for each time series
        for k, X in C.items():
            # Find the index maximizing the correlation
            # m =   list of indices of corr TS candidates AND
            #       already stored normally and different than itself
            m = {i: v for i, v in enumerate(X)
                 if (i in i_stored_1 and v >= threshold and k != i)
                 }
            m = dict(sorted(m.items(), key=lambda i: i[1], reverse=True))

            try:
                i_m = list(m.keys())[0]
            except:
                i_m = None
            if i_m is not None:  # store corr
                x = ts[i_m][w]
                y = ts[k][w]
                if k not in corr_coded_tricklets1.keys():
                    corr_coded_tricklets1[k] = {}
                corr_coded_tricklets1[k][w] = i_m

                # z = [v + shift_mean(x, y) for v in x]
                z = [v for v in x]
            else:  # store sparse
                if k not in atoms_coded_tricklets:
                    atoms_coded_tricklets[k] = {}
                atoms_coded_tricklets[k][w] = tricklets[k][w]
                # add k to the list of elements stored in sparse way
                i_stored_1.append(k)

    # for different time window
    for w_2 in range(len(result)):
        # create dictionary to keep indices
        A_2 = correlation_matrix2[w_2].values.tolist()
        B_2 = {i: A_2[i] for i in range(len(A_2))}
        # sort lines in a decent order by the sum of their elements
        C_2 = dict(sorted(B_2.items(), key=lambda i: sum(i[1]), reverse=True))

        i_stored_2 = []
        # for each time series
        for k_2, X_2 in C_2.items():
            # Find the index maximizing the correlation
            # m =   list of indices of corr TS candidates AND
            #       already stored normally and different than itself
            m_2 = {i_2: v_2 for i_2, v_2 in enumerate(X_2)
                 if (i_2 in i_stored_2 and v_2 >= threshold and k_2 != i_2)
                 }
            m_2 = dict(sorted(m_2.items(), key=lambda i: i[1], reverse=True))
            try:
                i_m_2 = list(m_2.keys())[0]
            except:
                i_m_2 = None
            if i_m_2 is not None:  # store corr
                if k_2 not in corr_coded_tricklets2.keys():
                    corr_coded_tricklets2[k_2] = {}
                corr_coded_tricklets2[k_2][w_2] = i_m_2
            else:  # store sparse
                if k_2 not in atoms_coded_tricklets:
                    atoms_coded_tricklets[k_2] = {}
                atoms_coded_tricklets[k_2][w_2] = tricklets[w_2][k_2]
                # add k to the list of elements stored in sparse way
                i_stored_2.append(k_2)
    return atoms_coded_tricklets,corr_coded_tricklets1,corr_coded_tricklets2


def shift_median(x, y):
    av_x = sum(x) / len(x)
    av_y = sum(y) / len(y)

    # print(av_y - av_x)
    # plt.plot(x)
    # plt.plot(y)
    # plt.plot([i + av_y - av_x for i in x])
    # plt.show()

    plt.show(block=True)

    return av_y - av_x


def shift_mean(x, y):
    import statistics

    av_x = statistics.median(x)
    av_y = statistics.median(y)

    # print(av_y - av_x)
    # plt.plot(x)
    # plt.plot(y)
    # plt.plot([i + av_y - av_x for i in x])
    # plt.show()

    plt.show(block=True)

    return av_y - av_x


def alpha_beta(x, y):
    try:

        acc = (y[-1] - y[0]) / (x[-1] - x[0])
        print(x)
        print(y)
        print(acc)

        plt.plot(x)
        plt.plot(y)
        plt.plot([i * acc + x[0] * (1 - acc) for i in x])
        plt.show()

        plt.show(block=True)

    except:
        acc = 0
    # print(acc, x[0]*(1-acc))
    return acc, x[0] * (1 - acc)


# import sys
# from numbers import Number
# from collections import Set, Mapping, deque
#
# try: # Python 2
#     zero_depth_bases = (basestring, Number, xrange, bytearray)
#     iteritems = 'iteritems'
# except NameError: # Python 3
#     zero_depth_bases = (str, bytes, Number, range, bytearray)
#     iteritems = 'items'
#
# def get_size(obj_0):
#     """Recursively iterate to sum size of object & members."""
#     _seen_ids = set()
#     def inner(obj):
#         obj_id = id(obj)
#         if obj_id in _seen_ids:
#             return 0
#         _seen_ids.add(obj_id)
#         size = sys.getsizeof(obj)
#         if isinstance(obj, zero_depth_bases):
#             pass # bypass remaining control flow and return
#         elif isinstance(obj, (tuple, list, Set, deque)):
#             size += sum(inner(i) for i in obj)
#         elif isinstance(obj, Mapping) or hasattr(obj, iteritems):
#             size += sum(inner(k) + inner(v) for k, v in getattr(obj, iteritems)())
#         # Check for custom object instances - may subclass above too
#         if hasattr(obj, '__dict__'):
#             size += inner(vars(obj))
#         if hasattr(obj, '__slots__'): # can have __slots__ with __dict__
#             size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
#         return size
#     return inner(obj_0)


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def compress_without_correlation(ts, Dictionary, nbAtoms, transform_algorithm):
    # Transforming test data into sparse respresentation using the transform algorithm
    print("Transforming test data into sparse respresentation without correlation ... ", end='')
    sparseData = sparse_code_without_correlation(ts, Dictionary, nbAtoms, transform_algorithm)
    print("done!")
    print("Reconstructing data...", end="")
    recons = reconstructDataMulti_without_correlation(sparseData, Dictionary)
    print("done!")
    errors = []
    for i in range(len(ts)):
        errors.append(calculate_RMSE(ts[i], recons[i]))
    return sparseData, errors


def compress_with_correlation1(ts, correlation_matrix, Dictionary, corr_threshold, nbAtoms, transform_algorithm):
    # Transforming test data into sparse respresentation using the omp algorithm
    print("Transforming test data into correlation-aware sparse representation ... ", end='')
    atoms_coded_tricklets, corr_coded_tricklets = sparse_code_with_correlation1(ts, correlation_matrix, Dictionary,
                                                                               nbAtoms, transform_algorithm,
                                                                               corr_threshold)
    print("done!")
    print("Reconstructing data with correlation...", end="")

    recons = reconstructDataMulti_with_correlation1(atoms_coded_tricklets, corr_coded_tricklets, Dictionary, ts)
    print("done!")

    import itertools
    errors = []
    result_before = []
    result_after = []
    for i in range(len(ts)):
        errors.append(calculate_RMSE(ts[i], recons[i]))
        result_before.append(list(itertools.chain.from_iterable(ts[i])))
        result_after.append(list(itertools.chain.from_iterable(recons[i])))
    result_after = pd.DataFrame(result_after)
    result_before = pd.DataFrame(result_before)
    result_before = result_before.T
    result_after = result_after.T
    print(result_before.shape)
    print(result_after.shape)
    print(result_before.head())
    print(result_after.head())
    result_after.to_csv('yoga_after.txt', float_format='%.6f', header=False, sep=' ', index=False)
    result_before.to_csv('yoga_before.txt', float_format='%.6f', header=False, sep=' ', index=False)

    return atoms_coded_tricklets, corr_coded_tricklets, errors



def compress_with_correlation2(ts, correlation_matrix1,correlation_matrix2, Dictionary, corr_threshold, nbAtoms, transform_algorithm):
    # Transforming test data into sparse representation using the omp algorithm
    print("Transforming test data into correlation-aware sparse representation ... ", end='')
    atoms_coded_tricklets,corr_coded_tricklets1,corr_coded_tricklets2 = sparse_code_with_correlation2(ts, correlation_matrix1,correlation_matrix2, Dictionary,
                                                                                nbAtoms, transform_algorithm,
                                                                                corr_threshold)
    print("done!")
    print("Reconstructing data with correlation...", end="")
    recons = reconstructDataMulti_with_correlation2(atoms_coded_tricklets, corr_coded_tricklets1,corr_coded_tricklets2,Dictionary, ts)
    print("done!")
    import itertools
    errors = []
    result_before = []
    result_after = []
    for i in range(len(ts)):
        errors.append(calculate_RMSE(ts[i], recons[i]))
        result_before.append(list(itertools.chain.from_iterable(ts[i])))
        result_after.append(list(itertools.chain.from_iterable(recons[i])))
    result_after = pd.DataFrame(result_after)
    result_before = pd.DataFrame(result_before)
    result_before = result_before.T
    result_after = result_after.T
    print(result_before.shape)
    print(result_after.shape)
    print(result_before.head())
    print(result_after.head())
    result_after.to_csv('yoga_after.txt', float_format='%.6f', header=False, sep=' ', index=False)
    result_before.to_csv('yoga_before.txt', float_format='%.6f', header=False, sep=' ', index=False)

    return atoms_coded_tricklets,corr_coded_tricklets1,corr_coded_tricklets2, errors







def normalized(ts):
    # pop = np.array([np.array(xi) for xi in ts])
    # return (pop - np.min(pop)) / (np.max(pop) - np.min(pop))
    from scipy import stats
    return stats.zscore(ts)


def calculate_RMSE(orig_sig, reconstructed_sig):
    orig_sig = normalized(orig_sig)
    reconstructed_sig = normalized(reconstructed_sig)
    return (np.square(np.array(orig_sig) - np.array(reconstructed_sig))).mean(axis=None)



def calculate_PRD(orig_sig, reconstructed_sig):
    orig_sig = normalized(orig_sig)
    reconstructed_sig = normalized(reconstructed_sig)
    num = np.sum((orig_sig - reconstructed_sig) ** 2)
    den = np.sum(orig_sig ** 2)
    PRD = np.sqrt(num / den)
    return PRD


def dataframeToTricklets(data, len_tricklet):
    ts = [[] for i in range(len(data.columns))]
    i = 1
    for column in data:
        # print(data[column].tolist())
        # print.print(list(chunks(data[column].tolist(), len_tricklet)))
        ts[data.columns.get_loc(column)].extend(chunks(data[column].tolist(), len_tricklet))
        # print(i)
        # i += 1
    return ts



def chunks(l, len_tricklet):
    """Yield successive n-sized chunks from l."""
    res = []
    for i in range(0, len(l), len_tricklet):
        if (len(l[i:i + len_tricklet]) == len_tricklet):
            res.append(l[i:i + len_tricklet])
    return res


def chunks2(l, len_tricklet):
    """Yield successive n-sized chunks from l."""
    res = []
    for i in range(0, len(l), len_tricklet):
        if (len(l[i:i + len_tricklet]) == len_tricklet):
            res.append(l[i:i + len_tricklet])

    ans = []
    i = 0
    while i < len(res):
        quant_data = scalar_quant(np.array(res[i]), max(res[i]))
        ans.append(quant_data.tolist())
        i += 1

    return ans



def reduce_mem_usage(df_data):
    start_mem=df_data.memory_usage().sum()/1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df_data.columns:
        col_type =df_data[col].dtype

        if col_type !=object:
            c_min=df_data[col].min()
            c_max=df_data[col].max()

            if str(col_type)[:3] =='int':
                if c_min >np.iinfo(np.int8).min and c_max <np.iinfo(np.int8).max:
                    df_data[col] = df_data[col].astype(np.int8)
                elif c_min>np.iinfo(np.int16).min and c_max<np.iinfo(np.int16).max:
                    df_data[col]=df_data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df_data[col]=df_data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df_data[col]=df_data[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max <np.finfo(np.float16).max:
                    df_data[col]=df_data[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df_data[col]=df_data[col].astype(np.float32)
                else:
                    df_data[col]=df_data[col].astype(np.float64)
        else:
            df_data[col]=df_data[col].astype('category')

    end_mem=df_data.memory_usage().sum()/1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df_data
