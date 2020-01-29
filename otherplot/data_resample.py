import pandas as pd
import numpy as np
import os

# output csv
# including cleaned data
def generate_csv(outputfilepath, df):
    df.to_csv(outputfilepath, sep=',', encoding='utf-8')



def resampleCSV(df,filepath):
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'],dayfirst=True)   ##use day first for GOV download csv
    df.set_index('TIMESTAMP',inplace=True)
    # df[df < 0] = 0
    # df.replace(0, np.nan).fillna(method='ffill')


    newcsv = df.resample('1H').mean()
    newcsv = newcsv.interpolate(method='linear', axis=0).bfill()
    print(newcsv.describe())
    # filedir, name = os.path.split(filepath)
    filename, file_extension = os.path.splitext(filepath)
    # outputcsv = os.path.join(filedir, name + '_resample' + '.csv')
    outputcsv = os.path.join(filename + '_resample' + '.csv')
    newcsv.to_csv(outputcsv)


filepath ='./data/burnett-river-trailer-quality-2015-all-forpca-norm.csv'


df = pd.read_csv(filepath)

resampleCSV(df,filepath)



# generate_csv(r"C:\Users\ZHA244\Coding\QLD\baffle_creek\baffle-creek-buoy-quality-2013-all-forpca-120min.csv",
#              df.iloc[::4])

