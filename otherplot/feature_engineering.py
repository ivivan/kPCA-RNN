import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, KernelPCA
from sklearn import preprocessing

# output csv
# including cleaned data
def generate_csv(outputfilepath, df):
    df.to_csv(outputfilepath, sep=',', encoding='utf-8')


# data input
filepath = './data/burnett-river-trailer-quality-2015-all-forpca.csv'

dataset = pd.read_csv(filepath)

#dataset_pca = dataset.drop(dataset.columns[1],axis=1)
dataset_pca = dataset.drop(['TIMESTAMP','DO_Sat','DO_mg','RECORD'],axis=1)
dataset_pca = dataset_pca.drop(dataset.columns[0],axis=1)
X = dataset_pca.as_matrix()

print(dataset_pca.head())



# Standardization
std_scale = preprocessing.StandardScaler().fit(X)
X_std = std_scale.transform(X)



# linear regression figure

print('create new df')



result_df = pd.DataFrame(X_std,columns=['Temp_degC','EC_uScm','pH','Turbidity_NTU','Chloraphylla_ugL'])
dataset['Temp_degC']=result_df['Temp_degC']
dataset['EC_uScm']=result_df['EC_uScm']
dataset['pH']=result_df['pH']
dataset['Turbidity_NTU']=result_df['Turbidity_NTU']
dataset['Chloraphylla_ugL']=result_df['Chloraphylla_ugL']
print(result_df.head())
print('end work')


# # create dataframe again
# print('create new df')
# result_df = pd.DataFrame(X[0:48],columns=['Temp_degC','EC_uScm','pH','Turbidity_NTU','Chloraphylla_ugL'])
# print(result_df.head())
# print('end work')

# kernel pca
# kpca = PCA().fit(X_std)
# gamma_range = np.logspace(-3, 1)
# for ga in gamma_range:
kpca = KernelPCA(n_components=5,kernel="rbf", gamma=0.02,fit_inverse_transform=True, degree=3).fit(X_std)
X_kpca_transformed = kpca.transform(X_std)

print('gamma:')
# print(ga)
print(kpca.lambdas_)
print(kpca.alphas_)


dataset['PC1'] = X_kpca_transformed[:,0]
dataset['PC2'] = X_kpca_transformed[:,1]
dataset['PC3'] = X_kpca_transformed[:,2]
dataset['PC4'] = X_kpca_transformed[:,3]
dataset['PC5'] = X_kpca_transformed[:,4]
# Results Analysis

explained_variance = np.var(X_kpca_transformed, axis=0)
explained_variance_ratio = explained_variance / np.sum(explained_variance)

print('step 1:')
print(explained_variance)
print('---------')
print('step 2:')
print(explained_variance_ratio)
print('----------')
print('step 3')
print(np.cumsum(explained_variance_ratio))
print('---------')

# drawing

# plt.figure()
#
# plt.scatter(X_kpca_transformed[:, 0], X_kpca_transformed[:, 1], c="red",
#             s=20, edgecolor='k')
# plt.title("Projection by PCA")
# plt.xlabel("1st principal component")
# plt.ylabel("2nd component")
# plt.show()






# # Generate New CSV

# outputpath = r'C:\Users\ZHA244\Coding\QLD\burnett_river\burnett-river-trailer-quality-2015-all-forpca-norm.csv'

# generate_csv(outputpath,dataset)