from sklearn.feature_selection import SelectKBest,f_regression,mutual_info_regression,chi2,variance_threshold,SelectFromModel,SelectPercentile
import scipy as sc
import pandas as pd


# data input
filepath = r'C:\Users\ZHA244\Coding\QLD\baffle_creek\baffle-creek-buoy-quality-2013-all-forpca.csv'

dataset = pd.read_csv(filepath)

Y = dataset['DO_mg'].values
dataset = dataset.drop(dataset.columns[0],axis=1)
dataset = dataset.drop(['TIMESTAMP','DO_Sat','DO_mg'],axis=1)
X = dataset.as_matrix()


# Pearsonr
for feature in range(5):
    print(sc.stats.pearsonr(dataset.iloc[:,feature].values,Y))


# Mutual Information
Select3=SelectKBest(mutual_info_regression,k=3)
X3=Select3.fit_transform(X,Y)
print(Select3.scores_)