import numpy as np
import pandas as pd
from sklearn import preprocessing
import otherplot.data_io_pca_multistep as dataio

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
    # get first num_sample rows for training set, with all columns
    dataset = dataset.iloc[start:start + num_sample, :]
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


def prepare_do():
    # Parameters
    model_params = {'TIMESTEPS': 24, 'N_FEATURES': 5}

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
        'scaler_four': scaler_ph,
        'scaler_five': scaler_chlo
    }

    # datafile
    filepath = './data/burnett-river-trailer-quality-2015-all-forpca-norm_resample.csv'
    # x, y = generate_data(filepath, 732,model_params['TIMESTEPS'], 0, 'train', scaler_dic)
    x, y = generate_data(filepath, 2928, model_params['TIMESTEPS'], 0, 'train',
                         scaler_dic)

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

    x_t, y_t = generate_data(filepath, 744 + model_params['TIMESTEPS'],
                             model_params['TIMESTEPS'],
                             2928 - model_params['TIMESTEPS'], 'test',
                             scaler_dic)
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
    y_test_ec = y_test_ec.reshape(-1, 1)
    y_test_temp = y_test_temp.reshape(-1, 1)

    y_test_do = scaler_do_y.transform(y_test_do)
    y_test_ec = scaler_ec_y.transform(y_test_ec)
    y_test_temp = scaler_temp_y.transform(y_test_temp)

    x_train = x_train.reshape(
        (x_train.shape[0],
         model_params['TIMESTEPS'] * model_params['N_FEATURES']))
    x_test = x_test.reshape(
        (x_test.shape[0],
         model_params['TIMESTEPS'] * model_params['N_FEATURES']))

    return x_train, x_test, y_train_do, y_test_do, scaler_do_y

    # print('x_train:{}'.format(x_train.shape))
    # print('x_test:{}'.format(x_test.shape))
    # print('y_train_do:{}'.format(y_train_do.shape))
    # print('x_train_do:{}'.format(y_test_do.shape))
