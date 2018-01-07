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
y=np.asarray(y)
#print(y[1])

#a=[1,2,3,4,5,6,7,8];
count=0
count1=0
count2=0;
# for i in y:
	# print (i)
	
	# if (count>0):
	
	
		# if (i>temp):
			# count1=count1+1
		# if (i<temp):
			# count2=count2+1
	# temp=i
	# count=count+1;
# print (count1)
# print (count)
# print (count2)	
	

ii=0;
count_pos=0;
count_neg=0;
cc=0
print(np.size(y))
while(ii<7990):
	while(y[ii+1]>y[ii]):
		count_pos=count_pos+1
		print('Positive count',ii)
		ii=ii+1;
		
	while(y[ii+1]<y[ii]):
		count_neg=count_neg+1;
		print('Negative count',ii)
		ii=ii+1;
		

print(ii)
print(count_neg)
print(count_pos)

plt.plot(x, y)
plt.xlabel('sample(n)')
plt.ylabel('voltage(V)')
plt.show()
