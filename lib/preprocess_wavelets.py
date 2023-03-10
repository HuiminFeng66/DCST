import numpy as np
import math
import pywt


# 封装成函数
def sgn(num):
    if (num > 0):
        return 1.0
    elif (num == 0):
        return 0.0
    else:
        return -1.0


def wavelet_noising(new_df):
    data = new_df
    # data = data.values.T.tolist()  # 将np.ndarray()转为列表
    data = data.values.tolist()  # 将np.ndarray()转为列表
    w = pywt.Wavelet('sym8')
    # [ca3, cd3, cd2, cd1] = pywt.wavedec(data, w, level=3)  # 分解波
    [ca5, cd5, cd4, cd3, cd2, cd1] = pywt.wavedec(data, w, level=5)  # 分解波

    length1 = len(cd1)
    length0 = len(data)

    Cd1 = np.array(cd1)
    abs_cd1 = np.abs(Cd1)
    median_cd1 = np.median(abs_cd1)

    sigma = (1.0 / 0.6745) * median_cd1
    lamda = sigma * math.sqrt(2.0 * math.log(float(length0), math.e))
    usecoeffs = []
    usecoeffs.append(ca5)  # 向列表末尾添加对象

    # 软硬阈值折中的方法
    a = 0.5

    for k in range(length1):
        if (abs(cd1[k].any()) >= lamda):
            cd1[k] = sgn(cd1[k].any()) * (abs(cd1[k].any()) - a * lamda)
        else:
            cd1[k] = 0.0

    length2 = len(cd2)
    for k in range(length2):
        if (abs(cd2[k].any()) >= lamda):
            cd2[k] = sgn(cd2[k].any()) * (abs(cd2[k].any()) - a * lamda)
        else:
            cd2[k] = 0.0

    length3 = len(cd3)
    for k in range(length3):
        if (abs(cd3[k].any()) >= lamda):
            cd3[k] = sgn(cd3[k].any()) * (abs(cd3[k].any()) - a * lamda)
        else:
            cd3[k] = 0.0

    length4 = len(cd4)
    for k in range(length4):
        if (abs(cd4[k].any()) >= lamda):
            cd4[k] = sgn(cd4[k].any()) * (abs(cd4[k].any()) - a * lamda)
        else:
            cd4[k] = 0.0

    length5 = len(cd5)
    for k in range(length5):
        if (abs(cd5[k].any()) >= lamda):
            cd5[k] = sgn(cd5[k].any()) * (abs(cd5[k].any()) - a * lamda)
        else:
            cd5[k] = 0.0

    usecoeffs.append(cd5)
    usecoeffs.append(cd4)
    usecoeffs.append(cd3)
    usecoeffs.append(cd2)
    usecoeffs.append(cd1)
    recoeffs = pywt.waverec(usecoeffs, w)
    return recoeffs


def denoise(data):
    data_denoising = wavelet_noising(data)  # 调用小波去噪函数
    return data_denoising