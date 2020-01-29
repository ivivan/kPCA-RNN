from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import scipy as sc
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator

sns.set(style="white")
sns.set_palette("husl")

filepath ='./data/burnett-river-trailer-quality-2015-all-forpca_resample.csv'
df = pd.read_csv(filepath)


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


fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(nrows=2, ncols=3,figsize=(10, 10))


sns.boxplot(y=df["Temp_degC"],ax=ax1).set(
    xlabel='Temperature',
    ylabel='$\u2103$'
)
# ax1.set_xlabel(fontsize=40)

sns.boxplot(y=df["EC_uScm"],ax=ax2).set(
    xlabel='EC',
    ylabel='uS $cm^{-1}$'
)

sns.boxplot(y=df["pH"],ax=ax3).set(
    xlabel='pH',
    ylabel=''
)

sns.boxplot(y=df["DO_mg"],ax=ax4).set(
    xlabel='DO',
    ylabel='mg $L^{-1}$'
)

sns.boxplot(y=df["Turbidity_NTU"],ax=ax5).set(
    xlabel='Turbidity',
    ylabel='NTU'
)

sns.boxplot(y=df["Chloraphylla_ugL"],ax=ax6).set(
    xlabel='Chl-a',
    ylabel='$\mu$g $L^{-1}$'
)

plt.tight_layout()
plt.show()