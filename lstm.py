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

x_train = x_train.reshape(x_train.shape[0],n_memory_steps,-1)
x_test = x_test.reshape(x_test.shape[0],n_memory_steps,-1)

print('x_train:{}'.format(x_train.shape))
print('x_test:{}'.format(x_test.shape))
print('y_train_do:{}'.format(y_train_do.shape))
print('x_train_do:{}'.format(y_test_do.shape))


## LSTM Model based on keras

input = Input(shape=(n_memory_steps, n_input_dim))  ## input_shape: shape of input data, (n_memory_steps, n_in_features)
x = Dropout(0.1)(input)
# x = GRU(units=40, return_sequences=True)(input)
# x = LSTM(units=30, return_sequences=True)(input)
# x = LSTM(units=40, return_sequences=True)(x)
x = LSTM(units=80, return_sequences=False)(x)
output = Dense(1, activation='linear')(x)



m = Model(inputs=input, outputs=output)

print('\nLSTM model summary:')
print(m.summary())

## Training and Evaluation

m.compile(optimizer='Adam', loss='mse')

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
checkpoint_callback = ModelCheckpoint('lstm31' + '.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# history = m.fit(x_train,y_train_do,batch_size=batch_size,epochs=epochs,validation_split=validation_split,shuffle=True,callbacks=[early_stopping_callback,checkpoint_callback])

# # predict data

trained_model = load_model('lstm31.h5')
print('\nTrained LSTM model summary:')
print(trained_model.summary())

y_pred = trained_model.predict(x_test)

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

timestepahead = 2
filepath = './data/burnett-river-trailer-quality-2015-all-forpca-norm_resample.csv'
axis_data = getdate_index(filepath,2928+timestepahead-1,742)

# line_upper = scaler_do_y.inverse_transform(y_test_do)[0:744]*1.1
# line_upper = line_upper.flatten()

# line_lower = scaler_do_y.inverse_transform(y_test_do)[0:744]*0.9
# line_lower = line_lower.flatten()
# print(line_upper.shape)

# plt.scatter(np.arange(744),error_array)
# plt.show()


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




fig = plt.figure(dpi=200)
# set up subplot grid
gridspec.GridSpec(3,1)

# large subplot
plt.subplot2grid((3,1), (0,0), rowspan=2)
# xfmt = mdates.DateFormatter('%Y-%m-%d')
# ax.xaxis.set_major_formatter(xfmt)

true_line, = plt.plot_date(axis_data, scaler_do_y.inverse_transform(y_test_do)[0:742], '-', lw=1, color=tableau20[2],
                           label='Observation')
predict_line, = plt.plot_date(axis_data, np.array(y_predicted)[0:742], '--',lw=1, color=tableau20[18],
                              label='Prediction')

plt.fill_between(axis_data, np.multiply(scaler_do_y.inverse_transform(y_test_do)[0:742],0.9).flatten(),np.multiply(scaler_do_y.inverse_transform(y_test_do)[0:742],1.1).flatten(),color=tableau20[15],interpolate=True,label='F1.1')

plt.legend(fontsize=12)
plt.title('3 Hour Ahead DO Concentration Prediction', fontsize=16)
plt.ylabel('DO (mg/l)', fontsize=14)


# small subplot 1
plt.subplot2grid((3,1), (2,0))
plt.plot_date(axis_data,error_array,'-',color=tableau20[8])
plt.ylabel('RMSE', fontsize=14)
plt.xlabel('Date', fontsize=14)
plt.gcf().autofmt_xdate()
# fit subplots and save fig
fig.tight_layout()



















# fig,(ax1,ax2) = plt.subplots(nrows=2, ncols=1)
# xfmt = mdates.DateFormatter('%Y-%m-%d')
# ax1.xaxis.set_major_formatter(xfmt)

# true_line, = ax1.plot_date(axis_data, scaler_do_y.inverse_transform(y_test_do)[0:744], '-', lw=1, color=tableau20[2],
#                            label='Observation')
# predict_line, = ax1.plot_date(axis_data, np.array(y_predicted)[0:744], '--',lw=1, color=tableau20[18],
#                               label='Prediction')

# ax1.fill_between(axis_data, np.multiply(scaler_do_y.inverse_transform(y_test_do)[0:744],0.9).flatten(),np.multiply(scaler_do_y.inverse_transform(y_test_do)[0:744],1.1).flatten(),color=tableau20[15],interpolate=True,label='F1.1')

# ax2.plot_date(axis_data,error_array,'-',color=tableau20[8])

# plt.legend(fontsize=12)
# plt.legend(handles=[true_line, predict_line],fontsize=12)
# plt.title('1 Hour Ahead DO Concentration Prediction', fontsize=16)
# plt.xlabel('Date', fontsize=14)
# plt.ylabel('DO (mg/l)', fontsize=14)
# plt.gcf().autofmt_xdate()
# plt.savefig(r'C:\Users\ZHA244\Pictures\paper2-figure\1hour-40-interval.png', dpi=260)



# # one prediction case
# plt.figure(figsize=(6,6))
# ax = plt.subplot(1, 1, 1)
# xfmt = mdates.DateFormatter('%Y-%m-%d %H:%M')
# ax.xaxis.set_major_formatter(xfmt)

# plt.plot_date(axis_data[0:12], scaler_do_y.inverse_transform(y_test_do)[0:12], '-', lw=1, color=tableau20[2],marker='.',
#                            label='Observation')

# plt.plot_date(axis_data[12:15], np.array([6.820,6.831,6.776]), '--',lw=1, color=tableau20[18],marker='*',
#                               label='Prediction')

# # plt.fill_between(axis_data,np.multiply(scaler_do_y.inverse_transform(y_test_do)[0:742],0.9).flatten(),np.multiply(scaler_do_y.inverse_transform(y_test_do)[0:742],1.1).flatten(),color=tableau20[15],interpolate=True,label='F1.1')

# # plt.legend(fontsize=12)
# plt.title('3 Steps Ahead DO Concentration Prediction', fontsize=16)
# plt.xlabel('Date', fontsize=14)
# plt.ylabel('DO (mg/l)', fontsize=14)
# plt.gcf().autofmt_xdate()
# # plt.savefig(r'C:\Users\ZHA244\Pictures\paper2-figure\onecase.png', figsize=(5,5))


plt.show()
