import pandas as pd
import numpy as np

array = np.array([[1,2,3],[77,78,79]])
print (array)

array_2 = np.append(array,[77,90,99])
print (array_2,'shape',np.shape(array_2))
 
array_3 = np.random.randint(4)
print (array_3,'shape',np.shape(array_3))
array_3=np.transpose(array_3)
print (array_3,'shape',np.shape(array_2))
print('Array 3_',array_3)
array_4=array_2*array_3
print('Matrice:',array_4)
import csv
csv_filename='eggs.csv'
with open(csv_filename, 'r') as fp:
    reader = csv.reader(fp)
    data = list(reader)
#array_data = np.array(data)

print(data)
