import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, KernelPCA
from sklearn import preprocessing




# data input
filepath = './data/baffle-creek-buoy-quality-2013-all-forpca.csv'

dataset = pd.read_csv(filepath)

Y = dataset['DO_mg'].values
dataset = dataset.drop(dataset.columns[0],axis=1)
dataset = dataset.drop(['TIMESTAMP','DO_Sat','DO_mg'],axis=1)
X = dataset.as_matrix()

# Standardization
std_scale = preprocessing.StandardScaler().fit(X)
X_std = std_scale.transform(X)


# create dataframe again
print('create new df')
result_df = pd.DataFrame(X[0:48],columns=['Temp_degC','EC_uScm','pH','Turbidity_NTU','Chloraphylla_ugL'])
print(result_df.head())
print('end work')

# kernel pca

kpca = KernelPCA(n_components=3,kernel="rbf", fit_inverse_transform=True, gamma=10).fit(X_std)
X_kpca_transformed = kpca.transform(X_std)
#print(len(kpca.lambdas_))

X_back = kpca.inverse_transform(X_kpca_transformed)

X_back_origin = std_scale.inverse_transform(X_back)

# create dataframe again
print('create new df after PCA')
result_df = pd.DataFrame(X_back_origin[0:48],columns=['Temp_degC','EC_uScm','pH','Turbidity_NTU','Chloraphylla_ugL'])
print(result_df.head())
print('end work PCA')
