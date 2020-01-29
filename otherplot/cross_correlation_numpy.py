from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import ccf

def getdate_index(filepath, start, num_predict):
    """
    :param filepath: same dataset file
    :param start: start now no. for prediction
    :param num_predict: how many predictions
    :return: the x axis datatime index for prediciton drawing
    """
    dataset = pd.read_csv(filepath)
    dataset = dataset.iloc[start:start + num_predict, :]
    dataset['TIMESTAMP'] = pd.to_datetime(dataset['TIMESTAMP'], dayfirst=True)

    return dataset['TIMESTAMP']


# Read Dataframe
# data input
filepath = r'C:\Users\ZHA244\Coding\QLD\burnett_river\burnett-river-trailer-quality-2015-all-forpca-norm.csv'

dataset = pd.read_csv(filepath)

Y = dataset['DO_mg'].values
Test = dataset['pH'].values
PC1 = dataset['PC1'].values
PC2 = dataset['PC2'].values

Total1 = np.append(PC1[:288],Y[288:336])
Total2 = np.append(PC2[:288],Y[288:336])


corr = ccf(PC1[:288], Y[0:288], unbiased=True)


fig, (ax_orig, ax_noise, ax_corr) = plt.subplots(3, 1)


ax_orig.plot(Y[0:228])
ax_orig.set_title('DO')


ax_noise.plot(PC1[:288])
ax_noise.set_title('PC1')
ax_corr.plot(corr)

ax_corr.set_title('Cross-correlated with rectangular pulse')
ax_orig.margins(0, 0.1)


fig.show()
plt.show()

