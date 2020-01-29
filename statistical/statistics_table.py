from pandas import Series
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
import pandas as pd
from matplotlib.ticker import AutoMinorLocator


filepath ='./data/burnett-river-trailer-quality-2015-all-forpca_resample.csv'

df = pd.read_csv(filepath, header=0)
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], dayfirst=True)



df1 = df.loc[(df['TIMESTAMP'] > '2015/9/1') & (df['TIMESTAMP'] < '2015/9/30')]
df2 = df.loc[(df['TIMESTAMP'] > '2015/9/2') & (df['TIMESTAMP'] < '2015/9/5')]

# plotting

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


fig, ax1 = plt.subplots(nrows=1, ncols=1,figsize=(15, 15))

ax1.plot_date(df1['TIMESTAMP'],df1['DO_mg'] , 'b-', color=tableau20[2],
                             label='DO')

ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.tick_params(axis='both', which='minor', labelsize=20)

ax1.get_xaxis().set_minor_locator(AutoMinorLocator())
ax1.grid(b=True, which='major', color='w', linewidth=1.5)
ax1.grid(b=True, which='major', color='w', linewidth=1.5)
plt.setp(ax1.get_xticklabels(), rotation=50, horizontalalignment='right')

plt.show()
