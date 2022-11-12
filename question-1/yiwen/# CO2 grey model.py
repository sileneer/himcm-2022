# CO2 grey model 
# install pandas, numpy, matplotlib 

import matplotlib.pyplot as plt 
import math 
import pandas as pd
import numpy as np 


df = pd.read_excel("2022_HiMCM_Data.xlsx")
data = pd.DataFrame(df)
# print(data.index)

x0 = []
t = data['PPM'][0:63]
# print(t)
for i in t: 
    x0.append(i)
# print ('x0',x0)

x1 = [x0[0]]
temp = x0[0]+x0[1]
x1.append(temp)
i=2 
while i < len(x0):
    temp += x0[i]
    x1.append(temp)
    i += 1
# print('x1',x1)

z1 = []
j = 1 
while j < len(x1):
    temp1 = (x1[j]+x1[j-1])/2
    z1.append(temp1)
    j += 1
# print ('z1',z1)

# 最小二乘法
Y = []
temp2 = 0 
while temp2 < len(x0)-1: 
    temp2 += 1 
    Y.append(x0[temp2])
Y = np.mat(Y).T 
Y.reshape(-1,1) # column matrix
# print ('Y',Y)

B = []
temp3 = 0 
while temp3 < len(z1): 
    B.append(-z1[temp3])
    temp3 += 1 
# print ('B',B)

B = np.mat(B) 
B.reshape(-1,1) 
B = B.T 
c = np.ones((len(B),1))
B = np.hstack((B,c)) 
# print ('c:',c)
# print ('new_B',B)

U = np.linalg.inv(B.T.dot(B)).dot(B.T).dot(Y) #calculate U
a = U[0]
u = U[1]
const = u/a 
# print ('U:',U)

# predictive model 
pred = [x0[0]]
temp4 = 1 
while temp4 < len(x0) +30: 
    pred.append((x0[0]-const)* math.exp(-a* temp4)+ const) 
    temp4 += 1 
# print ('predict',pred)

x_pred = [x0[0]]
k = 1 
while k < len(x0) +30: 
    x_pred.append(pred[k]- pred[k-1])
    k += 1
    
x0 = np.array(x0)
x_pred = np.array(x_pred) 
print ('x_pred:', x_pred)

'''
t1 = range (1959, 2022)
t2 = range (1959, 2052)


plt.plot(t1, x0, 'r-', label='true number')
plt.plot(t2, x_pred, 'b-', label='predict number')
plt.legend(loc='upper right')
plt.xlabel('years')
plt.ylabel('CO2 /PPM')
plt.title('The prediction of CO2 level')
plt.show()
'''









