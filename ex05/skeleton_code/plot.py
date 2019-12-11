import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

data_own = pd.read_csv('MyOwnTime.txt', sep=" ", header=None)
alpha = data_own[0].values
time_own = data_own[1].values

data_cblas = pd.read_csv('MyCblasTime.txt', sep=" ", header=None)
time_cblas = data_cblas[1].values

plt.figure()
plt.plot(alpha, time_own* 1e-6, c= 'b')
plt.plot(alpha, time_cblas* 1e-6, c= 'r')
plt.legend(['Own Implementation', 'CBLAS Implementation'])
plt.xlabel('alpha')
plt.ylabel('time [s]')
plt.savefig('result.png')
