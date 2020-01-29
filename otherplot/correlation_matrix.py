from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import scipy as sc
from matplotlib.patches import Rectangle

sns.set(style="white")

# Read Dataframe
# data input
filepath ='./data/burnett-river-trailer-quality-2015-all-forpca-norm.csv'

dataset = pd.read_csv(filepath)
Y = dataset['DO_mg'].values
d = dataset.drop(dataset.columns[0],axis=1)
d = d.drop(d.columns[0],axis=1)
d = d.drop(['TIMESTAMP','DO_Sat','DO_mg','RECORD'],axis=1)

#rename for drawing
d.rename(columns={'Temp_degC':'Temperature', 'Turbidity_NTU':'Turbidity', 'EC_uScm':'EC', 'Chloraphylla_ugL':'Chl-a'}, inplace=True)


# sns.set(font_scale=1.1)

# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(10, 9))
ax.tick_params(labelsize=14)

# Generate a custom diverging colormap
cmap = sns.diverging_palette(240, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
hm = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .5})



ax.add_patch(Rectangle((4,6), 1, 1, fill=False, edgecolor='blue', lw=3, linestyle='--'))
ax.add_patch(Rectangle((3,5), 1, 1, fill=False, edgecolor='blue', lw=3,linestyle='--'))
ax.add_patch(Rectangle((0,5), 1, 1, fill=False, edgecolor='blue', lw=3,linestyle='--'))
ax.add_patch(Rectangle((2,5), 1, 1, fill=False, edgecolor='blue', lw=3,linestyle='--'))
ax.add_patch(Rectangle((1,6), 1, 1, fill=False, edgecolor='blue', lw=3,linestyle='--'))
ax.add_patch(Rectangle((1,7), 1, 1, fill=False, edgecolor='green', lw=3))
ax.add_patch(Rectangle((2,7), 1, 1, fill=False, edgecolor='green', lw=3))
ax.add_patch(Rectangle((3,8), 1, 1, fill=False, edgecolor='green', lw=3))
ax.add_patch(Rectangle((1,5), 1, 1, fill=False, edgecolor='green', lw=3))

plt.show()