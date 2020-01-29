import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import seaborn as sns;
sns.set(color_codes=True)

n_bins = 50

sample = np.random.uniform(0, 1, 50)
ecdf = ECDF(sample)

x = np.linspace(min(sample), max(sample))
# y = ecdf(x)
# plt.step(x, y)

fig, ax = plt.subplots(figsize=(8, 4))

# plot the cumulative histogram
n, bins, patches = ax.hist(sample, n_bins, normed=1, histtype='step',
                           cumulative=True, label='Empirical')



plt.show()


