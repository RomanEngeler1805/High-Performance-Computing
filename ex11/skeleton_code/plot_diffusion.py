import numpy as np
from matplotlib import pyplot as plt

# 
myarray = np.fromfile('diagnostics_openmp.dat', dtype=float)

print(myarray.shape)

#
plt.figure()
plt.plot(myarray)
plt.show()