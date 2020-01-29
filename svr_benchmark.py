# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from datetime import datetime, date
from sklearn import preprocessing
from sklearn.svm import SVR
import time
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from sklearn.model_selection import TimeSeriesSplit
from data_process import data_io_pca_multistep as dataio
from sklearn.externals import joblib


np.set_printoptions(threshold=np.inf)

# Prepare data

def generate_data(filepath, num_sample, timestamp, start, mode, scalardic):
    """
    :param filepath: data set for the model
    :param start: start row for training set, for training, start=0
    :param num_sample: how many samples used for training set, in this case, 2928 samples from 1st Oct-30th Nov, two month
    :param timestamp: timestamp used for LSTM
    :return: training set, train_x and train_y
    """
    dataset = pd.read_csv(filepath)
    dataset = dataset.iloc[start:start + num_sample, :]  # get first num_sample rows for training set, with all columns
    dataset['TIMESTAMP'] = pd.to_datetime(dataset['TIMESTAMP'], dayfirst=True)


    set_x, set_y = dataio.load_csvdata(dataset, timestamp, mode, scalardic)
    return set_x, set_y


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


# Parameters
model_params = {
    'TIMESTEPS': 12,
    'N_FEATURES':3
    # 'RNN_LAYERS': [{'num_units': 400}],
    # # 'RNN_LAYERS': [{'num_units': 60, 'keep_prob': 0.75},{'num_units': 120, 'keep_prob': 0.75},{'num_units': 60, 'keep_prob': 0.75}],
    # 'DENSE_LAYERS': None,
    # 'TRAINING_STEPS': 15000,
    # 'PRINT_STEPS': 50,
    # 'BATCH_SIZE': 100
}


# Scale x data (training set) to 0 mean and unit standard deviation.
scaler_do = preprocessing.StandardScaler()
scaler_ec = preprocessing.StandardScaler()
scaler_temp = preprocessing.StandardScaler()
scaler_ph = preprocessing.StandardScaler()
scaler_chlo = preprocessing.StandardScaler()

scaler_dic = {
    'scaler_one': scaler_do,
    'scaler_two': scaler_ec,
    'scaler_three': scaler_temp,
}

# datafile
# filepath = r'C:\Users\ZHA244\Coding\QLD\baffle_creek\baffle-creek-buoy-quality-2013-all-forpca-120min.csv'
filepath = r'C:\Users\ZHA244\Coding\QLD\burnett_river\burnett-river-trailer-quality-2015-all-forpca-norm.csv'
# x, y = generate_data(filepath, 732,model_params['TIMESTEPS'], 0, 'train', scaler_dic)
x, y = generate_data(filepath, 2928, model_params['TIMESTEPS'], 0, 'train', scaler_dic)

scaler_dic['scaler_one'] = x['scalerone']
scaler_dic['scaler_two'] = x['scalertwo']
scaler_dic['scaler_three'] = x['scalerthree']

# Training set, three train y for multiple tasks training
x_train = x['train']
y_train_do = y['trainyone']
y_train_ec = y['trainytwo']
y_train_temp = y['trainythree']

# x_t, y_t = generate_data(filepath, 372+12, model_params['TIMESTEPS'], 732-12, 'test', scaler_dic)  # testing set for 240 prediction (5 days)
x_t, y_t = generate_data(filepath, 744+model_params['TIMESTEPS'], model_params['TIMESTEPS'], 2928-model_params['TIMESTEPS'], 'test', scaler_dic)

# Testing set, three test y for multiple tasks testing
x_test = x_t['train']
y_test_do = y_t['trainyone']
y_test_ec = y_t['trainytwo']
y_test_temp = y_t['trainythree']

# Scale y data to 0 mean and unit standard deviation
scaler_do_y = preprocessing.StandardScaler()
scaler_ec_y = preprocessing.StandardScaler()
scaler_temp_y = preprocessing.StandardScaler()

y_train_do = y_train_do.reshape(-1, 1)
y_train_ec = y_train_ec.reshape(-1, 1)
y_train_temp = y_train_temp.reshape(-1, 1)


y_train_do = scaler_do_y.fit_transform(y_train_do)
y_train_ec = scaler_ec_y.fit_transform(y_train_ec)
y_train_temp = scaler_temp_y.fit_transform(y_train_temp)

y_test_do = y_test_do.reshape(-1, 1)
y_test_ec = y_test_ec.reshape(-1,1)
y_test_temp = y_test_temp.reshape(-1,1)

y_test_do = scaler_do_y.transform(y_test_do)
y_test_ec = scaler_ec_y.transform(y_test_ec)
y_test_temp = scaler_temp_y.transform(y_test_temp)


x_train = x_train.reshape((x_train.shape[0], model_params['TIMESTEPS']* model_params['N_FEATURES']))
x_test = x_test.reshape((x_test.shape[0],model_params['TIMESTEPS']* model_params['N_FEATURES']))



# x_train = x.get('train').reshape(2925, 3)
# y_train = y.get('train')
#
#
# x_test = x_t.get('train').reshape(1488, 3)
# y_test = y_t.get('train')
#
# y_train = y_train.reshape(-1, 1)
# y_test = y_test.reshape(-1,1)




#
#
y_train_do = np.ravel(y_train_do)
y_test_do = np.ravel(y_test_do)

index = 0



X_index = []
prediction_value = []

# for index in range(len(x_train)):
#     # Search for best parameters
#     regressor = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
#                              param_grid={"C": [1e0, 1e1, 1e2, 1e3],
#                                          "gamma": np.logspace(-3, 3, 20)})
#
#     regressor.fit(x_train, y_train)
#     print(regressor.best_estimator_)
#     print(regressor.best_params_)
#
#     print("prediction ------")
#     y_pred = scaler_y.inverse_transform(regressor.predict(x_test))
#     # X_index.append(index+time_steps)
#     prediction_value.append(y_pred.item(0))


print(x_train.shape)
print(y_train_do.shape)


    # Search for best parameters
regressor = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                             param_grid={"C": [1e0, 1e1, 1e2],
                                         "gamma": np.logspace(-1, 1, 10)})

regressor.fit(x_train, y_train_do)
print(regressor.best_estimator_)
print(regressor.best_params_)

# s = joblib.dump(regressor,'svr.pkl')

# regressor = joblib.load('svr.pkl')

# print("prediction ------")
# y_pred = scaler_y.inverse_transform(regressor.predict(x_test))

for index in range(len(x_test)):
    y_pred = scaler_do_y.inverse_transform(regressor.predict(x_test[index].reshape(-1,36)))
    prediction_value.append(y_pred.item(0))

    # X_index.append(index+time_steps)
# prediction_value.append(y_pred.item(0))

print("Sec:")
print(prediction_value)

print("True Value:")
print(scaler_do_y.inverse_transform(y_test_do))
# True_value = dataset.iloc[X_index[0]:X_index[-1]+1,]
# print(True_value['Turbidity_NTU'].values)
print("------------")
print("Predict Value:")
print(prediction_value)
print("-------------")
print("RMSE:")
mse = mean_squared_error(scaler_do_y.inverse_transform(y_test_do), prediction_value)
rmse = sqrt(mse)
print(rmse)

print("--------")
print("MAE:")
mae = mean_absolute_error(scaler_do_y.inverse_transform(y_test_do), prediction_value)
print(mae)
print("---------")
print("R2:")
r2 = r2_score(scaler_do_y.inverse_transform(y_test_do), prediction_value)
print(r2)


k=0
for i,j in zip(np.array(prediction_value),scaler_do_y.inverse_transform(y_test_do)):
    if 0.9 < i/j < 1.1:
        k += 1

f_one = k/len(np.array(prediction_value))

print('F1.1:{0:f}'.format(f_one))

# painting
timestepahead = 3
axis_data = getdate_index(filepath,732+timestepahead-1,5)
#
ax=plt.gca()
xfmt = mdates.DateFormatter('%Y-%m-%d %H:%M')
ax.xaxis.set_major_formatter(xfmt)

true_line, = plt.plot_date(axis_data,scaler_do_y.inverse_transform(y_test_do)[0:5],'b-', color='blue', label = 'True Value')
predict_line, = plt.plot_date(axis_data,prediction_value[0:5],'b-', color='Red', label = 'Prediction Value')

plt.legend(handles=[true_line, predict_line])
plt.gcf().autofmt_xdate()
plt.show()