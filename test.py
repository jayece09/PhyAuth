import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

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
    print (s,d0[i])
    print ("")
	
test=np.asarray(p1);
my_df = pd.DataFrame(test)
my_df.to_csv('pdata_pass.csv', index=False, header=False)
#print (my_df)

test1=np.asarray(z1);
my_df1 = pd.DataFrame(test1)
my_df1.to_csv('desk_pass.csv', index=False, header=False)
print (my_df1)
# print (test);
# test.tofile('foo.csv',sep=',',format='%10.5f');
plt.plot(p1)
plt.plot(z1)
plt.show()
