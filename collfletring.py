import pandas as pd
import numpy as np

users = {"Angelica": {"Blues Traveler": 3.5, "Broken Bells": 2.0,
"Norah Jones": 4.5, "Phoenix": 5.0,
"Slightly Stoopid": 1.5,
"The Strokes": 2.5, "Vampire Weekend": 2.0},
"Bill": {"Blues Traveler": 2.0, "Broken Bells": 3.5,
"Deadmau5": 4.0, "Phoenix": 2.0,
"Slightly Stoopid": 3.5, "Vampire Weekend": 3.0},
"Chan": {"Blues Traveler": 5.0, "Broken Bells": 1.0,
"Deadmau5": 1.0, "Norah Jones": 3.0,
"Phoenix": 5, "Slightly Stoopid": 1.0},
"Dan": {"Blues Traveler": 3.0, "Broken Bells": 4.0,
"Deadmau5": 4.5, "Phoenix": 3.0,
"Slightly Stoopid": 4.5, "The Strokes": 4.0,
"Vampire Weekend": 2.0},
"Hailey": {"Broken Bells": 4.0, "Deadmau5": 1.0,
"Norah Jones": 4.0, "The Strokes": 4.0,
"Vampire Weekend": 1.0},
"Jordyn": {"Broken Bells": 4.5, "Deadmau5": 4.0, "Norah Jones": 5.0,
"Phoenix": 5.0, "Slightly Stoopid": 4.5,
"The Strokes": 4.0, "Vampire Weekend": 4.0},
"Sam": {"Blues Traveler": 5.0, "Broken Bells": 2.0,
"Norah Jones": 3.0, "Phoenix": 5.0,
"Slightly Stoopid": 4.0, "The Strokes": 5.0},
"Veronica": {"Blues Traveler": 3.0, "Norah Jones": 5.0,
"Phoenix": 4.0, "Slightly Stoopid": 2.5,
"The Strokes": 3.0}}

dfu = pd.DataFrame(users)

def manhattan(rating1,rating2):
    return sum(abs((rating1.dropna()-rating2.dropna()).dropna()))

def computeNearestNeighbor(username,users):
    users1 = users.drop([username],axis=1)
    return sorted(users1.columns.map(lambda user: (minkowski(users1[user],users[username],2),user)).tolist(),key=lambda at:at[0])

def recommend(username, users):
    nbs =[ v for k,v in computeNearestNeighbor(username, users)[:3]]
    x= users[nbs][np.isnan(dfu[username])]
    return x[x.mean(axis=1) > 2].index.values

def minkowski(rating1, rating2, r):
    distance = sum(pow(abs((rating1.dropna()-rating2.dropna()).dropna()),r))
    return pow(distance,1/r)

def pearson(rating1, rating2):
    commonr=pd.concat([rating1, rating2]).dropna().index.get_duplicates()
    n = len(commonr)
    denominator = math.sqrt((rating1[commonr]**2).sum()-((rating1[commonr].sum()**2)/n)) * math.sqrt((rating2[commonr]**2).sum()-((rating2[commonr].sum()**2)/n))
    return 0 if denominator == 0 else ((rating1[commonr]*rating2[commonr]).sum()-(rating1[commonr].sum()*rating2[commonr].sum()/n))/denominator
