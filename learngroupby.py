import pandas as pd
import numpy as np

df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
                           'foo', 'bar', 'foo', 'foo'],
                   'B' : ['one', 'one', 'two', 'three',
                          'two', 'two', 'one', 'three'],
                   'C' : np.random.randn(8), 'D' : np.random.randn(8)})

def get_letter_type(letter):
        print(letter)
        if letter.lower() in 'aeiou':
            return 'vowel'
        else:
            return 'consonant'


grouped = df.groupby(['A', 'B'],as_index=False)
print(grouped.aggregate(np.sum))
print(grouped.sum())
groupbySingleCol = df.groupby(['A'])
groupbymulcols = df.groupby(['A', 'B'])
print(groupbySingleCol.size())
print(groupbymulcols.size())
#print(groupbySingleCol.describe())
#print(groupbymulcols.describe())
#print(groupbySingleCol['C'].agg([np.sum,np.mean,np.std]))
#print(groupbymulcols['C'].agg([np.sum,np.mean,np.std]))
print(groupbySingleCol.agg([np.sum,np.mean,np.std]))
print(groupbymulcols.agg([np.sum,np.mean,np.std]))
