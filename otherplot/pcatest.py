import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles

for i in range(5):
    print(i)

x = np.array([3,4,5,6])
y = np.array([[1,2],[2,3],[3,4],[4,5]])

x = x[1:]
y = y[:len(x)]

print(x)
print(y)