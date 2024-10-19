import pandas as pd

def converti (file_path):
   
   df = pd.read_csv(file_path) 
   data = pd.DataFrame(df)
   return data

path='eggs.csv'
dataC = converti(path)
print (dataC)

