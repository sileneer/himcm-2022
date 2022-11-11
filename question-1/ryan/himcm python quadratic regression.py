
#main code
import csv
file = open("himcm_data_csv.csv")
csvreader = csv.reader(file)
header = next(csvreader)
rows = []
no_of_rows = 63
for row in csvreader:
    rows.append(row)
'''
print(rows)
'''

train = rows[:50]
test = rows[50:]
yearlist = []
ppmlist = []

import numpy as np
from matplotlib import pyplot as plt
for i in rows:
    #updating
    yearlist.append(float(i[0]) - 1958)
    ppmlist.append(float(i[1]))
#numpy conversion
npyear = np.array(yearlist)
np_ppm = np.array(ppmlist)
'''
print(npyear)
print(np_ppm)
'''
#scatter plot

y = np.polyfit(npyear, np_ppm, 2)
print(y)
plt.plot(npyear, y[0]*(npyear**2) + y[1]*(npyear) + y[2])
plt.scatter(npyear, np_ppm)
plt.show()
print(y[0]*(92**2) + y[1]*(92) + y[2])

file.close()
