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
	
def quant():
	ii=0;
	c_pos=[];
	c_neg=[];
	count_pos=0;
	count_neg=0;
	cc_pos=0;
	cc_neg=0;
	key_pos=[];
	key_neg=[];
	key=[]
	print(np.size(y))
	while(ii<7997):
		while(y[ii+1]>y[ii]) and (ii+1<7998):
			count_pos=count_pos+1
			print('Positive count',ii)
			#if (ii+1>7997):
				#break
			ii=ii+1;
			cc_pos=cc_pos+1;
		key_pos.append(bin(cc_pos));
		c_pos.append(cc_pos);
		key.append(bin(cc_pos))
		cc_pos=0;
		
		while(y[ii+1]<y[ii]) and (ii+1<7998):
			count_neg=count_neg+1;
			print('Negative count',ii)
			ii=ii+1;
			cc_neg=cc_neg+1;
		key_neg.append(bin(cc_neg));
		c_neg.append(cc_neg);
		key.append(bin(cc_neg))
		cc_neg=0;
			
	print('Increasing order=', c_pos);
	print('Decreasing Order=', c_neg);
	print('Binary Pos Key=', key_pos)
	print('Binary Neg key=', key_neg)
	print('Final Key=', key)
	print(ii)
	print(count_neg)
	print(count_pos)
quant()
plt.plot(x, y)
plt.xlabel('sample(n)')
plt.ylabel('voltage(V)')
plt.show()
