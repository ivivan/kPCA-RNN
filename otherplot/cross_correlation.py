from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




# Read Dataframe
# data input
filepath = r'C:\Users\ZHA244\Coding\QLD\burnett_river\burnett-river-trailer-quality-2015-all-forpca-norm.csv'

dataset = pd.read_csv(filepath)
Y = dataset['DO_mg'].values
Test = dataset['pH'].values
PC1 = dataset['PC1'].values
PC2 = dataset['PC2'].values

print(Y[:10])
print('----')
print(PC1[:10])


corr = signal.correlate(PC1[:288], Y[288:336], mode='valid')
print(corr)

fig, (ax_orig, ax_noise, ax_corr) = plt.subplots(3, 1, sharex=True)
ax_orig.plot(Y[288:336])
ax_orig.set_title('DO')
ax_noise.plot(PC1[:288])
ax_noise.set_title('PC1')
ax_corr.plot(corr)

ax_corr.set_title('Cross-correlated with rectangular pulse')
ax_orig.margins(0, 0.1)
fig.tight_layout()
fig.show()
plt.show()
