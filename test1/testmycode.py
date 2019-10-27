import numpy as np


a = np.asarray([1, 1, 2, 3, 5])
b = np.asarray([2, 3, 2, 3, 1])
print(np.sum((a==b)+0))