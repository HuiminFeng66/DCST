import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler



data=np.load('../datasets/HT_Sensor_dataset_combined.npy')
print(data.shape)   # (928991, 8)
data=data[:550000,:]
print(data.shape)    # (550000, 8)
np.savetxt('../datasets/UCRArchive_2018/gas/HT_Sensor_dataset_combined.csv', data, delimiter='\t')


