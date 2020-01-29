from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import scipy as sc
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import variation

filepath = './data/burnett-river-trailer-quality-2015-all-forpca.csv'
df = pd.read_csv(filepath)

print(variation(df['Temp_degC']))
print(variation(df['EC_uScm']))
print(variation(df['pH']))
print(variation(df['DO_mg']))
print(variation(df['Turbidity_NTU']))
print(variation(df['Chloraphylla_ugL']))

