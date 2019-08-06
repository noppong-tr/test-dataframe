import numpy as np
import pandas as pd

# # A structured array
# my_array = np.ones(3, dtype=([('foo', int), ('bar', float)]))
# # Print the structured array
# print(my_array['foo'])

# # A record array
# my_array2 = my_array.view(np.recarray)
# # Print the record array
# print(my_array2.foo)

# # Example use DataFrame 
# data = np.array([['','Col1','Col2'],
#                 ['Row1',1,2],
#                 ['Row2',3,4]])
                
# print(pd.DataFrame(data=data[1:,1:],
#                   index=data[1:,0],
#                   columns=data[0,1:]))

# # Take a 2D array as input to your DataFrame 
# my_2darray = np.array([[1, 2, 3], [4, 5, 6]])
# print(pd.DataFrame(my_2darray))

# # Take a dictionary as input to your DataFrame 
# my_dict = {1: ['1', '3'], 2: ['1', '2'], 3: ['2', '4']}
# print(pd.DataFrame(my_dict))

# # Take a DataFrame as input to your DataFrame 
# my_df = pd.DataFrame(data=[4,5,6,7], index=range(0,4), columns=['A'])
# print(pd.DataFrame(my_df))

# # Take a Series as input to your DataFrame
# my_series = pd.Series({"United Kingdom":"London", "India":"New Delhi", "United States":"Washington", "Belgium":"Brussels"})
# print(pd.DataFrame(my_series))

# # Example DataFrame use numpy 
# df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6]]))

# # Use the `shape` property => number(row, col)
# print(df.shape)

# # Or use the `len()` function with the `index` property => number(row)
# print(len(df.index))

# Setting value DataFrame 
# data_test = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(pd.DataFrame(data_test))
# data_new_from = pd.DataFrame(data=data_test, index=range(0, 3), columns=['A','B','C'])
# print(data_new_from)
# Set DataFrame in df
# df = data_new_from

# # Using `iloc[]` => `iloc[index_row][index]`
# print(df.iloc[0][0])

# # Using `loc[]` => `loc[index_row][colunm_name]`
# print(df.loc[0]['A'])

# # Using `at[]` => `at[index_row,colunm_name]` 
# print(df.at[0,'A'])

# # Using `iat[]` => `iat[index_row,index_column]`
# print(df.iat[0,0])

# # Use `iloc[]` to select row `0`
# print(df.iloc[0])

# # Use `loc[]` to select column `'A'`
# print(df.loc[:,'A'])

# my_dict = {'A': [1,4,7], 'B': [2,5,8], 'C': [3,6,9]}
# df = pd.DataFrame(my_dict)

# # Print out your DataFrame `df` to check it out
# print(df)

# # Set 'C' as the index of your DataFrame
# df.set_index('C')
# print(df.set_index('C'))

# df = pd.DataFrame(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), index= [2, 'A', 4], columns=[48, 49, 50])
# print(df)

# # Pass `2` to `loc` use row name 
# print(df.loc['A'])

# # Pass `2` to `iloc` use index of row
# print(df.iloc[0])

# # Pass `2` to `ix` use index of row
# print(df.ix[1])

# df = pd.DataFrame(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), index= [2.5, 12.6, 4.8], columns=[48, 49, 50])

# # There's no index labeled `2`, so you will change the index at position `2`
# df.ix[2] = [60, 50, 40]
# print(df)
# # This will make an index labeled `2` and add the new values
# df.loc[2] = [11, 12, 13]
# print(df)
# # Check out the weird index of your dataframe
# print(df)
# # Use `reset_index()` to reset the values. 
# df_reset = df.reset_index(level=0, drop=True)
# # Print `df_reset`
# print(df_reset)


# df = pd.DataFrame(data=np.array([[1, 1, 2], [3, 2, 4]]), index=range(2), columns=[1, 2, 3])
# # Study the DataFrame `df`
# print(df)
# # Append a column to `df`
# df.loc[:, 4] = pd.Series(['5', '6'], index=df.index)
# # Print out `df` again to see the changes
# print(df)


# df = pd.DataFrame(data=np.array([[1,2,3], [4,5,6], [7,8,9]]), index=range(3), columns=['A', 'B', 'C'])
# # Check out the DataFrame `df`
# print(df)
# # Drop the column with label 'A'                  
# df.drop('A', axis=1, inplace=True)
# print("Drop the column with label 'A'")
# print(df)

# df_2 = pd.DataFrame(data=np.array([[1,2,3], [4,5,6], [7,8,9]]), index=range(3), columns=['A', 'B', 'C'])
# Drop the column at position 1
# df_2.drop(df_2.columns[1], axis=1)
# print("Drop the column at position 1")
# print(df_2)

# df = pd.DataFrame(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [40, 50, 60], [23, 35, 37]]), 
#                   index= [2.5, 12.6, 4.8, 4.8, 2.5], 
#                   columns=[48, 49, 50])

# df = pd.DataFrame(data=np.array([[1, 2, 3, 4], [4, 5, 6, 5], [7, 8, 9, 6], [40, 50, 60, 7], [23, 35, 37, 23]]), 
#                   index= [2.5, 12.6, 4.8, 4.8, 2.5], 
#                   columns=[48, 49, 50, 50])

# # df.reset_index().drop_duplicates(subset='index', keep='last').set_index('index')
# # Check out your DataFrame `df`
# print(df)

# # Drop the duplicates in `df`
# z = df.drop_duplicates([48], keep='last')
# print(z)


# df = pd.DataFrame(data=np.array([[1,2,3], [4,5,6], [7,8,9]]), index=range(3), columns=['A', 'B', 'C'])
# # Check out the DataFrame `df`
# print(df)

# # Drop the index at position 1
# print(df.drop(df.index[1]))

# # Check out your DataFrame `df`
# print(df)

# # Define the new names of your columns
# newcols = {
#     'A': 'new_column_1', 
#     'B': 'new_column_2', 
#     'C': 'new_column_3'
# }

# # Use `rename()` to rename your columns
# df.rename(columns=newcols, inplace=True)
# print(df)

# # Rename your index
# print(df.rename(index={1: 'a'}))


# # Check out your DataFrame
# print(df)

# # Delete unwanted parts from the strings in the `result` column
# df['result'] = df['result'].map(lambda x: x.lstrip('+-').rstrip('aAbBcC'))

# # Check out the result again
# print(df)
#  class test result
#     0     1    2    +3b
#     1     4    5    -6B
#     2     7    8    +9A
#       class test result
#     0     1    2      3
#     1     4    5      6
#     2     7    8      9


# # Inspect your DataFrame `df`
# print(df)
#       Age PlusOne             Ticket
#     0  34       0           23:44:55
#     1  22       0           66:77:88
#     2  19       1  43:68:05 56:34:12

# # Split out the two values in the third row
# # Make it a Series
# # Stack the values
# ticket_series = df['Ticket'].str.split(' ').apply(pd.Series, 1).stack()

# # Get rid of the stack:
# # Drop the level to line up with the DataFrame
# ticket_series.index = ticket_series.index.droplevel(-1)

# # Make your `ticket_series` a dataframe 
# ticketdf = pd.DataFrame(ticket_series)

# # Delete the `Ticket` column from your DataFrame
# del df['Ticket']

# # Join the `ticketdf` DataFrame to `df`
# df.join(ticketdf)



#     0  0    23:44:55
#     1  0    66:77:88
#     2  0    43:68:05

#        1    56:34:12

#       Age PlusOne
#     0  34       0
#     1  22       0
#     2  19       1

# # Check out the new `df`
# print(df)


# df = pd.DataFrame(np.nan, index=[0,1,2,3], columns=['A'])
# print(df)

# df = pd.DataFrame(index=range(0,4),columns=['A'], dtype='float')
# print(df)

# Does Pandas Recognize Dates When Importing Data
# pd.read_csv('yourFile', parse_dates=True)

# # or this option:
# pd.read_csv('yourFile', parse_dates=['columnName'])

# dateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

# # Which makes your read command:
# pd.read_csv(infile, parse_dates=['columnName'], date_parser=dateparse)

# # Or combine two columns into a single DateTime column
# pd.read_csv(infile, parse_dates={'datetime': ['date', 'time']}, date_parser=dateparse)


# products = pd.DataFrame({'category': ['Cleaning', 'Cleaning', 'Entertainment', 'Entertainment', 'Tech', 'Tech'],
#                         'store': ['Walmart', 'Dia', 'Walmart', 'Fnac', 'Dia','Walmart'],
#                         'price':[11.42, 23.50, 19.99, 15.95, 55.75, 111.55],
#                         'testscore': [4, 3, 5, 7, 5, 8]})
# print(products)

# # Use `pivot()` to pivot the DataFrame
# pivot_products = products.pivot(index='category', columns='store', values='price')

# # Check out the result
# print(pivot_products)


# # Construct the DataFrame
# products = pd.DataFrame({'category': ['Cleaning', 'Cleaning', 'Entertainment', 'Entertainment', 'Tech', 'Tech'],
#                         'store': ['Walmart', 'Dia', 'Walmart', 'Fnac', 'Dia','Walmart'],
#                         'price':[11.42, 23.50, 19.99, 15.95, 55.75, 111.55],
#                         'testscore': [4, 3, 5, 7, 5, 8]})

# # Use `pivot()` to pivot your DataFrame
# pivot_products = products.pivot(index='category', columns='store')

# # Check out the results
# print(pivot_products)


# # Your DataFrame
# products = pd.DataFrame({'category': ['Cleaning', 'Cleaning', 'Entertainment', 'Entertainment', 'Tech', 'Tech'],
#                         'store': ['Walmart', 'Dia', 'Walmart', 'Fnac', 'Dia','Walmart'],
#                         'price':[11.42, 23.50, 19.99, 15.95, 19.99, 111.55],
#                         'testscore': [4, 3, 5, 7, 5, 8]})

# # Pivot your `products` DataFrame with `pivot_table()`
# pivot_products = products.pivot_table(index='category', columns='store', values='price', aggfunc='mean')

# # Check out the results
# print(pivot_products)


# # The `people` DataFrame
# people = pd.DataFrame({'FirstName' : ['John', 'Jane'],
#                        'LastName' : ['Doe', 'Austen'],
#                        'BloodType' : ['A-', 'B+'],
#                        'Weight' : [90, 64]})

# # Use `melt()` on the `people` DataFrame
# print(pd.melt(people, id_vars=['FirstName', 'LastName'], var_name='measurements'))


# df = pd.DataFrame(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['A', 'B', 'C'])

# for index, row in df.iterrows() :
#     print(row['A'], row['B'])

#Using Series
s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)

# Create datetime index and labeled columns
dates = pd.date_range('20130101', periods=6)
print(dates)

