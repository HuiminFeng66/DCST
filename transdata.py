import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# data=np.load('../datasets/UCRArchive_2018/electricity/electricity.npy')
# print(data.shape)  # (26304, 321)
# np.savetxt('../datasets/UCRArchive_2018/electricity/electricity.csv', data, delimiter='\t')


# data=np.load('../datasets/UCRArchive_2018/temperature/temperature01.npy')
# print(data.shape)  # (295719, 18)
# np.savetxt('../datasets/UCRArchive_2018/temperature/temperature01.csv', data, delimiter='\t')



# data=np.load('../datasets/UCRArchive_2018/acc/Watch_accelerometer_combined.npy')
# print(data.shape)
# data=data.T
# data=data[:200000,:]
# print(data.shape)   # (200000, 3)
# np.savetxt('../datasets/UCRArchive_2018/acc/Watch_accelerometer_combined.csv', data, delimiter='\t')




# data=np.load('../datasets/UCRArchive_2018/gyr/Watch_gyroscope_combined.npy')
# print(data.shape)   # (3, 3205431)
# data=data.T
# data=data[:1000000,:]
# print(data.shape)   # (1000000, 3)
# np.savetxt('../datasets/UCRArchive_2018/gyr/Watch_gyroscope_combined.csv', data, delimiter='\t')


data=np.load('../datasets/UCRArchive_2018/gas/HT_Sensor_dataset_combined.npy')
print(data.shape)   # (928991, 8)
data=data[:550000,:]
print(data.shape)    # (550000, 8)
np.savetxt('../datasets/UCRArchive_2018/gas/HT_Sensor_dataset_combined.csv', data, delimiter='\t')



# data=np.load('../datasets/UCRArchive_2018/exchangerate/exchangerate.npy')
# print(data.shape)   # (7588, 8)
# np.savetxt('../datasets/UCRArchive_2018/exchangerate/exchangerate.csv', data, delimiter='\t')

