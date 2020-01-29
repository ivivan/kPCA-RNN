from pandas import Series
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
import pandas as pd





filepath = './data/burnett-river-trailer-quality-2015-all-forpca_resample.csv'

df = pd.read_csv(filepath, header=0)

# pacf
plot_pacf(df['DO_mg'], lags=48, title='Partial Autocorrelation of DO')

# acf
# plot_acf(df['DO_mg'], lags=96)
pyplot.xlabel('Time Steps',fontsize=12)
pyplot.ylabel('Correlation Coefficient',fontsize=12)
pyplot.show()