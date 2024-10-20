import pandas as pd

def CsvToDf (file_path):
   
   df = pd.read_csv(file_path) 
   data = pd.DataFrame(df)
   return data

path='eggs.csv'
dataC = CsvToDf(path)



