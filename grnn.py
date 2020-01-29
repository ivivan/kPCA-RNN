import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Dropout, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from otherplot.prepare_data import prepare_do

from pyGRNN import GRNN
from neupy import algorithms

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

    return dataset['TIMESTAMP'].dt.to_pydatetime()


## Hyperparameters

n_input_dim = 5  # no. of features for input time series
n_output_dim = 1  # no. of features for output time series
n_memory_steps = 24  # length of input
n_forcast_steps = 1  # length of output
# train_test_split = 0.8  # protion as train set
validation_split = 0.1  # protion as validation set
batch_size = 2  # batch size for training
epochs = 10000  # epochs for training

## Data Processing

# load data

x_train, x_test, y_train_do, y_test_do, scaler_do_y = prepare_do()


print('x_train:{}'.format(x_train.shape))
print('x_test:{}'.format(x_test.shape))
print('y_train_do:{}'.format(y_train_do.shape))
print('x_train_do:{}'.format(y_test_do.shape))





## GRNN Model


nw = algorithms.GRNN(std=1, verbose=False)
nw.train(x_train, y_train_do)

y_pred = nw.predict(x_test)

print(y_pred)


# # Example 2: use Anisotropic GRNN with Limited-Memory BFGS algorithm to select the optimal bandwidths
# AGRNN = GRNN()
# AGRNN.fit(x_train, y_train_do.ravel())
# sigma=AGRNN.sigma 
# y_pred = AGRNN.predict(X_test)



# x_test_ori = data_scaler.inverse_transform(x_test.reshape(-1, n_memory_steps))
# y_test_ori = scaler_do_y.inverse_transform(y_test_do)
y_predicted = scaler_do_y.inverse_transform(y_pred)


# Performance metrics for all the regression targets.
# Score with sklearn.
print('Performance for DO prediciton:')
score_sklearn = mean_squared_error(scaler_do_y.inverse_transform(y_test_do), y_predicted)
print('RMSE (sklearn): {0:f}'.format(sqrt(score_sklearn)))
print("--------")
mae = mean_absolute_error(scaler_do_y.inverse_transform(y_test_do), y_predicted)
print("MAE (sklearn):{0:f}".format(mae))
print("---------")
r2 = r2_score(scaler_do_y.inverse_transform(y_test_do), y_predicted)
print("R2 (sklearn):{0:f}".format(r2))



# F1.1
k=0
for i,j in zip(np.array(y_predicted),scaler_do_y.inverse_transform(y_test_do)):
    if 0.9 < i/j < 1.1:
        k += 1

f_one = k/len(np.array(y_predicted))

print('F1.1:{0:f}'.format(f_one))

# mse for each value

error_array = []

for i in range(0,y_test_do.shape[0]):
    error_array.append(mean_squared_error(scaler_do_y.inverse_transform(y_test_do[i]), y_predicted[i]))

print(len(error_array))

