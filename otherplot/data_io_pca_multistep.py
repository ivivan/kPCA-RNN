import numpy as np
import pandas as pd
from sklearn import preprocessing


def rnn_data(data, time_steps, labels=False):
    """
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> labels == False [[1, 2], [2, 3], [3, 4]] #Data frame for input with 2 timesteps
        -> labels == True [3, 4, 5] # labels for predicting the next timestep
    """
    rnn_df = []
    for i in range(len(data) - time_steps):
        if labels:
            try:
                rnn_df.append(data.iloc[i + time_steps].as_matrix())
            except AttributeError:
                rnn_df.append(data.iloc[i + time_steps])
        else:
            data_ = data.iloc[i: i + time_steps].as_matrix()
            rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])

    return np.array(rnn_df, dtype=np.float64)


def change_predictive_interval(train,label,predictiveinterval):
    label = label[predictiveinterval-1:]
    train = train[:len(label)]

    return train,label


def load_csvdata(rawdata, time_steps,mode,scalardic):
    data = rawdata
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    train_x = rnn_data(data['DO_mg'], time_steps, labels=False)
    train_y = rnn_data(data['DO_mg'], time_steps, labels=True)

    train_x_two = rnn_data(data['PC1'], time_steps, labels=False)
    train_y_two = rnn_data(data['PC1'], time_steps, labels=True)

    train_x_three = rnn_data(data['PC2'], time_steps, labels=False)
    train_y_three = rnn_data(data['PC2'], time_steps, labels=True)

    train_x_four = rnn_data(data['PC3'], time_steps, labels=False)
    train_y_four = rnn_data(data['PC3'], time_steps, labels=True)

    train_x_five = rnn_data(data['PC4'], time_steps, labels=False)
    train_y_five = rnn_data(data['PC4'], time_steps, labels=True)


    prediction_horizon = 1


    #change predictive time interval
    train_x,train_y = change_predictive_interval(train_x,train_y,prediction_horizon) # two timesteps, 1 hour prediction
    train_x_two, train_y_two = change_predictive_interval(train_x_two, train_y_two, prediction_horizon)
    train_x_three, train_y_three = change_predictive_interval(train_x_three, train_y_three, prediction_horizon)
    train_x_four, train_y_four = change_predictive_interval(train_x_four, train_y_four, prediction_horizon)
    train_x_five, train_y_five = change_predictive_interval(train_x_five, train_y_five, prediction_horizon)

    train_x = np.squeeze(train_x)
    train_x_two = np.squeeze(train_x_two)
    train_x_three = np.squeeze(train_x_three)
    train_x_four = np.squeeze(train_x_four)
    train_x_five = np.squeeze(train_x_five)


    # Scale data (training set) to 0 mean and unit standard deviation.
    if(mode=='train'):
        scaler_do = scalardic['scaler_one']
        scaler_ec = scalardic['scaler_two']
        scaler_temp = scalardic['scaler_three']


        train_x = scaler_do.fit_transform(train_x)



    elif (mode=='test'):
        scaler_do = scalardic['scaler_one']
        scaler_ec = scalardic['scaler_two']
        scaler_temp = scalardic['scaler_three']

        train_x = scaler_do.transform(train_x)



    all_train = np.stack((train_x, train_x_two, train_x_three, train_x_four, train_x_five), axis=-1)
    all_train = all_train.reshape(-1,time_steps*5)

    return dict(train=all_train,scalerone=scaler_do,scalertwo=scaler_ec,scalerthree=scaler_temp), dict(trainyone=train_y,trainytwo=train_y_two,trainythree=train_y_three)
