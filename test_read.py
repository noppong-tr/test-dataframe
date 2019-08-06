import numpy as np
import pandas as pd

# print ("Pandas version",pd.__version__)

# Read CSV to DataFrame and Not set header
# csv_df = pd.read_csv('https://people.sc.fsu.edu/~jburkardt/data/csv/addresses.csv', header=None)
# print(csv_df)
# print(type(csv_df))

# Read CSV to DataFrame and set header
# csv_df = pd.read_csv('https://people.sc.fsu.edu/~jburkardt/data/csv/addresses.csv', 
#                     names=['first name', 'last name', 'address', 'citynm', 'city code', 'citynb'])
# print(csv_df)

# Sometimes reading CSV for Excel need encoding
# csv_df = pd.read_csv('https://people.sc.fsu.edu/~jburkardt/data/csv/addresses.csv',encoding = "ISO-8859-1")

# Read CSV to DataFrame (have a header)
csv_df = pd.read_csv('https://people.sc.fsu.edu/~jburkardt/data/csv/ford_escort.csv')
print(csv_df)

# set_df = pd.DataFrame(csv_df)
# print(set_df)

# df = pd.DataFrame(data=set_df, columns=['first name', 'last name', 'address', 'citynm', 'city code', 'citynb'], index=range(len(csv_df.index)))
# df = pd.DataFrame(csv_df, index=range(len(csv_df.index)))
# print(df)
