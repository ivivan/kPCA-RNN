from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import preprocessing
from math import sqrt
from data_process import data_io_pca_multistep as dataio
from forsecondpaper.rnn_support import lstm_model

tf.logging.set_verbosity(tf.logging.INFO)

# Parameters
# Parameters
model_params = {
    'TIMESTEPS': 12,
    'N_FEATURES': 3,
    'RNN_LAYERS': [{'num_units': 40}],
    # 'RNN_LAYERS': [{'num_units': 30, 'keep_prob': 0.25},{'num_units': 100, 'keep_prob': 0.25},{'num_units': 30, 'keep_prob': 0.25}],
    'DENSE_LAYERS': None,
    'TRAINING_STEPS': 1,
    'PRINT_STEPS': 50,
    'BATCH_SIZE': 10
}


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

    return dataset['TIMESTAMP'].dt.to_pydatetime()


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
    # 'scaler_four': scaler_ph,
    # 'scaler_five': scaler_chlo
}

# datafile
#filepath = r'C:\Users\ZHA244\Coding\QLD\baffle_creek\baffle-creek-buoy-quality-2013-all-forpca-120min.csv'
#filepath = r'C:\Users\ZHA244\Dropbox\Coding\QLD\burnett_river\burnett-river-trailer-quality-2015-all-forpca-norm.csv'
filepath = r'C:\Users\ZHA244\Dropbox\Coding\QLD\burnett_river\burnett-river-trailer-quality-2015-all-forpca-norm-1h.csv'
#filepath = r'C:\Users\ZHA244\Dropbox\Coding\QLD\burnett_river\burnett-river-trailer-quality-2015-all-forpca-norm-daily.csv'
#x, y = generate_data(filepath, 4419, model_params['TIMESTEPS'], 0, 'train', scaler_dic)
x, y = generate_data(filepath, 2928, model_params['TIMESTEPS'], 0, 'train', scaler_dic)
#x, y = generate_data(filepath, 122, model_params['TIMESTEPS'], 0, 'train', scaler_dic)
scaler_dic['scaler_one'] = x['scalerone']
scaler_dic['scaler_two'] = x['scalertwo']
scaler_dic['scaler_three'] = x['scalerthree']
# scaler_dic['scaler_four'] = x['scalerfour']
# scaler_dic['scaler_five'] = x['scalerfive']

# Training set, three train y for multiple tasks training
x_train = x['train']
y_train_do = y['trainyone']
y_train_ec = y['trainytwo']
y_train_temp = y['trainythree']

#x_t, y_t = generate_data(filepath, 1489+12, model_params['TIMESTEPS'], 5860-12, 'test', scaler_dic)  # testing set for 240 prediction (5 days)
x_t, y_t = generate_data(filepath, 744+model_params['TIMESTEPS'], model_params['TIMESTEPS'], 2928-model_params['TIMESTEPS'], 'test', scaler_dic)
#x_t, y_t = generate_data(filepath, 31+model_params['TIMESTEPS'], model_params['TIMESTEPS'], 122-model_params['TIMESTEPS'], 'test', scaler_dic)
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

# Prepare Regressor for multiple tasks learning
regressor = tf.estimator.Estimator(model_fn=lstm_model, model_dir=r'C:\Users\ZHA244\Coding\tensorflow\logs',
                                   params=model_params)

x_train = x_train.reshape((x_train.shape[0], model_params['TIMESTEPS'], model_params['N_FEATURES']))
x_test = x_test.reshape((x_test.shape[0], model_params['TIMESTEPS'], model_params['N_FEATURES']))
# Train.
# label_dict_train = {
#     'y_train_ntu': y_train_do,
#     'y_train_ec': y_train_ec,
#     'y_train_temp': y_train_temp
# }
#
# label_dict_test = {
#     'y_test_ntu': y_test_do,
#     'y_test_ec': y_test_ec,
#     'y_test_temp': y_test_temp
# }

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    # x={'x': x_train}, y=y_train_do, batch_size=model_params['BATCH_SIZE'], num_epochs=None, shuffle=False)
    x = {'x': x_train}, y = y_train_do, batch_size = model_params['BATCH_SIZE'], num_epochs = None, shuffle = True)
regressor.train(input_fn=train_input_fn, steps=model_params['TRAINING_STEPS'])

# Predict.


test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': x_test}, y=y_test_do, num_epochs=1, batch_size=model_params['BATCH_SIZE'], shuffle=False)
predictions = regressor.predict(input_fn=test_input_fn)

# results = list(predictions)

# y_predicted = np.array(list(scaler_ntu_y.inverse_transform(p) for p in predictions))


# y_predicted_ntu = [scaler_do_y.inverse_transform(p['predictions'])[0] for p in results]
# y_predicted_ec = [scaler_ec_y.inverse_transform(p['prediction_ec'])[0] for p in results]
# y_predicted_temp = [scaler_temp_y.inverse_transform(p['prediction_temp'])[0] for p in results]

# y_predicted = np.array(list(scaler_do_y.inverse_transform(p['predictions']) for p in predictions))

y_predicted = np.array(list(scaler_do_y.inverse_transform(p) for p in predictions))



# predicted = np.asmatrix(list(predictions),dtype = np.float64) #,as_iterable=False))


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

# Score with tensorflow
rmse_tf = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(scaler_do_y.inverse_transform(y_test_do), y_predicted))))
with tf.Session() as sess:
    print('RMSE (tensorflow):{0:f}'.format(sess.run(rmse_tf)))

# F1.1
k=0
for i,j in zip(np.array(y_predicted),scaler_do_y.inverse_transform(y_test_do)):
    if 0.9 < i/j < 1.1:
        k += 1

f_one = k/len(np.array(y_predicted))

print('F1.1:{0:f}'.format(f_one))




# Drawing


# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

timestepahead = 1
axis_data = getdate_index(filepath,2928+timestepahead-1,742)



# ax = plt.subplot(1, 1, 1)
# xfmt = mdates.DateFormatter('%Y-%m-%d')
# ax.xaxis.set_major_formatter(xfmt)
#
# true_line, = plt.plot_date(axis_data, scaler_do_y.inverse_transform(y_test_do)[0:742], '-', lw=1, color=tableau20[2],
#                            label='Observation')
#
# predict_line, = plt.plot_date(axis_data, np.array(y_predicted)[0:742], '--',lw=1, color=tableau20[18],
#                               label='Prediction')
#
# plt.fill_between(axis_data,np.multiply(scaler_do_y.inverse_transform(y_test_do)[0:742],0.9).flatten(),np.multiply(scaler_do_y.inverse_transform(y_test_do)[0:742],1.1).flatten(),color=tableau20[15],interpolate=True,label='F1.1')
#
# plt.legend(fontsize=12)
# plt.title('3 Hour Ahead DO Concentration Prediction', fontsize=16)
# plt.xlabel('Date', fontsize=14)
# plt.ylabel('DO (mg/l)', fontsize=14)
# plt.gcf().autofmt_xdate()
# plt.savefig(r'C:\Users\ZHA244\Pictures\paper2-figure\3hour-20-interval.png', dpi=260)



# ax = plt.subplot(1, 1, 1)
# xfmt = mdates.DateFormatter('%Y-%m-%d')
# ax.xaxis.set_major_formatter(xfmt)
#
# true_line, = plt.plot_date(axis_data, scaler_do_y.inverse_transform(y_test_do)[0:336], '-', lw=1, color=tableau20[2],
#                            label='Observation')
# true_line_upper, = plt.plot_date(axis_data, scaler_do_y.inverse_transform(y_test_do)[0:336]*1.1, lw=1, color=tableau20[2],
#                            label='UPPER')
# true_line_lower, = plt.plot_date(axis_data, scaler_do_y.inverse_transform(y_test_do)[0:336]*0.9, lw=1, color=tableau20[2],
#                            label='LOWER')
# predict_line, = plt.plot_date(axis_data, np.array(y_predicted)[0:336], '--',lw=1, color=tableau20[18],
#                               label='Prediction')
#
# plt.fill_between(axis_data, true_line_upper,true_line_lower,color='m',alpha=.5)
#
# plt.legend(handles=[true_line, predict_line,true_line_upper,true_line_lower],fontsize=12)
# plt.title('1 Hour Ahead DO Concentration Prediction', fontsize=16)
# plt.xlabel('Date', fontsize=14)
# plt.ylabel('DO (mg/l)', fontsize=14)
# plt.gcf().autofmt_xdate()
# plt.savefig(r'C:\Users\ZHA244\Pictures\paper2-figure\1hour-40-interval.png', dpi=260)



# one prediction case
plt.figure(figsize=(6,6))
ax = plt.subplot(1, 1, 1)
xfmt = mdates.DateFormatter('%Y-%m-%d %H:%M')
ax.xaxis.set_major_formatter(xfmt)

plt.plot_date(axis_data[0:12], scaler_do_y.inverse_transform(y_test_do)[0:12], '-', lw=1, color=tableau20[2],marker='.',
                           label='Observation')

plt.plot_date(axis_data[12:15], np.array([6.820,6.831,6.776]), '--',lw=1, color=tableau20[18],marker='*',
                              label='Prediction')

# plt.fill_between(axis_data,np.multiply(scaler_do_y.inverse_transform(y_test_do)[0:742],0.9).flatten(),np.multiply(scaler_do_y.inverse_transform(y_test_do)[0:742],1.1).flatten(),color=tableau20[15],interpolate=True,label='F1.1')

# plt.legend(fontsize=12)
plt.title('3 Steps Ahead DO Concentration Prediction', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('DO (mg/l)', fontsize=14)
plt.gcf().autofmt_xdate()
plt.savefig(r'C:\Users\ZHA244\Pictures\paper2-figure\onecase.png', figsize=(5,5))






plt.show()






print(scaler_do_y.inverse_transform(y_test_do)[0:20])
print(np.array(y_predicted)[0:20])


