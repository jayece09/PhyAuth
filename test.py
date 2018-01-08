import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from math import factorial

file1=open("desk.csv", "r")
file2=open("pdata.csv", "r")
reader1=csv.reader(file1)
reader2=csv.reader(file2)
d0=[]
d1=[]

z1=[]
p1=[]

def near(mylist,value):
    array = np.asarray(mylist)
    idx = (np.abs(array-value)).argmin()
    return idx

for line in reader1:
    s=line[0]
    s = s[:0] + s[8:]
    d0.append(float(s))
    d1.append(line[1])
	
	
for line in reader2:
    s=line[0]
    s = s[:0] + s[8:]
    i=near(d0,float(s)) 	
    z1.append(d1[i])
    p1.append(line[1])
    #print (s,d0[i])
    #print ("")

	
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    
    from math import factorial
    
    # try:
        # window_size = np.abs(np.int(window_size))
        # order = np.abs(np.int(order))
    # except ValueError, msg:
        # raise ValueError("window_size and order have to be of type int")
    # if window_size % 2 != 1 or window_size < 1:
        # raise TypeError("window_size size must be a positive odd number")
    # if window_size < order + 2:
        # raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

	



test=np.asarray(p1);
test1=np.asarray(z1);

my_df = pd.DataFrame(test)
my_df.to_csv('pdata_pass.csv', index=False, header=False)

test = list(map(int, test));
test1 = list(map(int, test1));
test=np.asarray(test);
test1=np.asarray(test1);
#print (test)
yhat = savitzky_golay(test, 200, 5) # window size 51, polynomial order 3 # In 8_Jan 2018 version it was 200,5
yhat1=savitzky_golay(test1, 200, 5)


## Algorithm: If increaing note number of samples it increases to: N:- binary(N) would be the key.
count_inc=0;
count_dec=0;


## Normalisation and Shape correlation
m=np.mean(yhat);
m1=np.mean(yhat1);
marray=np.ones(2000)
m1array=np.ones(2000)
marray=marray*m
m1array=m1array*m1
#print(marray)
# arr=[];
# ik=0;
# for ii in yhat:
	# if ii> (m+ 0.3*np.var(yhat))
		# arr(ik)=1
		# else
		# arr(ik)=0;
			

#print (test)
plt.plot(yhat)
plt.plot(test, color='red')
plt.plot(yhat1,color='black')
plt.plot(test1,color='yellow')
plt.plot(marray)
plt.plot(m1array)
plt.show()
#Zf = savitzky_golay( test, window_size=29, order=4)

## Arrays to compare: yhat and yhat1: Cross-Correlation after normalisation
yhat_m=np.array(yhat)/np.amax(yhat);

yhat1_m=np.array(yhat1)/np.amax(yhat1);

corr=np.correlate(yhat_m,yhat1_m)/2000; # Normalise with number of samples

print('The correlation is ', corr)

plt.plot(yhat_m)
plt.plot(yhat1_m)
plt.show()

####################################################################
#.............................QUANTIZATION..........................

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
print(np.size(yhat))
while(ii<1997):
	while(yhat[ii+1]>yhat[ii]) and (ii+1<1998):
		count_pos=count_pos+1
		#print('Positive count',ii)
		#if (ii+1>7997):
			#break
		ii=ii+1;
		cc_pos=cc_pos+1;
	key_pos.append(bin(cc_pos));
	c_pos.append(cc_pos);
	key.append(bin(cc_pos))
	cc_pos=0;
	
	while(yhat[ii+1]<yhat[ii]) and (ii+1<1998):
		count_neg=count_neg+1;
		#print('Negative count',ii)
		ii=ii+1;
		cc_neg=cc_neg+1;
	key_neg.append(bin(cc_neg));
	c_neg.append(cc_neg);
	key.append(bin(cc_neg))
	cc_neg=0;
		
print('Increasing order=', c_pos);
print('Decreasing Order=', c_neg);
#print('Binary Pos Key=', key_pos)
#print('Binary Neg key=', key_neg)
#print('Final Key=', key)
#print(ii)
print(count_neg)
print(count_pos)

########### Yhat1
ii1=0;
c_pos1=[];
c_neg1=[];
count_pos1=0;
count_neg1=0;
cc_pos1=0;
cc_neg1=0;
key_pos1=[];
key_neg1=[];
key1=[]
print('Size of Yhat1=', np.size(yhat1))
while(ii1<1997):
	while(yhat1[ii1+1]>=yhat1[ii1]) and (ii1+1<1999):
		count_pos1=count_pos1+1
		#print('Positive count of Phone',ii1)
		#if (ii+1>7997):
			#break
		ii1=ii1+1;
		cc_pos1=cc_pos1+1;
	key_pos1.append(bin(cc_pos1));
	c_pos1.append(cc_pos1);
	key1.append(bin(cc_pos1))
	cc_pos1=0;
	
	while(yhat1[ii1+1]<=yhat1[ii1]) and (ii1+1<1999):
		count_neg1=count_neg1+1;
		#print('Negative count of Phone',ii1)
		ii1=ii1+1;
		cc_neg1=cc_neg1+1;
	key_neg1.append(bin(cc_neg1));
	c_neg1.append(cc_neg1);
	key1.append(bin(cc_neg1))
	cc_neg1=0;
		
print('Increasing order Phone=', c_pos1);
print('Decreasing Order Phone=', c_neg1);
#print('Binary Pos Key Phone=', key_pos1)
#print('Binary Neg key Phone=', key_neg1)
#print('Final Key Phone=', key1)
#print(ii1)
print(count_neg1)
print(count_pos1)


######################################################




test1=np.asarray(z1);
my_df1 = pd.DataFrame(test1)
my_df1.to_csv('desk_pass.csv', index=False, header=False)
#print (my_df1)
# print (test);
# test.tofile('foo.csv',sep=',',format='%10.5f');
plt.plot(p1)
plt.plot(z1)
plt.show()
