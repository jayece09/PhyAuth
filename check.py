import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from math import factorial


Fs = 8000
f = 5
sample = 8000
x = np.arange(sample)
y = np.sin(2 * np.pi * f * x / Fs)


a=[1,2,3,4,5,6,7,8];
count=0
count1=0
for i in a:
	print (i)
	
	if (count>0):
		if (i>temp):
			count1=count1+1
	temp=i
	count=count+1;
print (count1)
print (count)
plt.plot(x, y)
plt.xlabel('sample(n)')
plt.ylabel('voltage(V)')
plt.show()
