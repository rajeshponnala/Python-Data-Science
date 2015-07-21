import pandas as pd
import numpy as np

headers = ['name','title','department','salary']
chicago = pd.read_csv('city-of-chicago-salaries.csv',header=False,names=headers,converters={'salary': lambda x: float(x.replace('$',''))})
by_dept = chicago.groupby('department')
print(by_dept.count().head())
print('\n')
print(by_dept.size().head())
print('\n')
print(by_dept.sum()[20:25])
print('\n')
print(by_dept.mean()[20:25])
print('\n')
print(by_dept.median()[20:25])
print('\n')
print(by_dept.title.nunique().order(ascending=False)[:5])

def ranker(df):
    df['dept_rank'] = np.arange(len(df))+1
    return df

chicago.sort('salary', ascending=False, inplace=True)
chicago = chicago.groupby('department').apply(ranker)
print(chicago[chicago.dept_rank == 1].head(7))
