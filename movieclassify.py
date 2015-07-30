import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user',sep='|', names=u_cols)

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols)

m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(5),encoding='latin-1')

movies_ratings = pd.merge(movies,ratings)
lens = pd.merge(movies_ratings,users)

l = len(sorted(lens['user_id'].unique()[:100],reverse=False))
z = sorted(lens["user_id"].unique()[:100])
df4= pd.DataFrame(np.zeros((l,l)),index=z,columns=z)


def minkowski(rating1, rating2, r):
    distance = sum(pow(abs((rating1.dropna()-rating2.dropna()).dropna()),r))
    return pow(distance,1/r)


def get_movie_distance_table(data):
    newdata = data[['user_id','rating','movie_id']]
    newdata.set_index('movie_id',inplace=True)
    l = len(sorted(newdata['user_id'].unique()[:100],reverse=False))
    z = sorted(data["user_id"].unique()[:100])
    df4= pd.DataFrame(np.zeros((l,l)),index=z,columns=z)
    for a in df4.columns:
         for b in df4.index:
            # df4.ix[b][a] = pearson(newData[newData["user_id"]==b]["rating"],newData[newData["user_id"]==a]["rating"])
            #  df4.ix[b][a] = manhattan(newdata[newdata["user_id"]==b]["rating"],newdata[newdata["user_id"]==a]["rating"])
               df4.ix[b][a] = minkowski(newdata[newdata["user_id"]==b]["rating"],newdata[newdata["user_id"]==a]["rating"],2)
    return df4

def loadData():
  df4 = pd.read_csv('moviedist.csv')
  return df4

def refresh_data():
    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv('ml-100k/u.user',sep='|', names=u_cols)
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols)
    m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
    movies = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(5),encoding='latin-1')
    movies_ratings = pd.merge(movies,ratings)
    lens = pd.merge(movies_ratings,users)
    get_movie_distance_table(lens).to_csv('moviedist.csv')


def nearestNeighbours(user,df4):
    return df4.ix[user]

def predictUserMovieRating(user,movie,df4):
    y = nearestNeighbours(user,df4)
    y= y[1:].index.map(lambda x: int(x))
    return lens["rating"][(lens["user_id"].isin(y)) & (lens["movie_id"] == movie)].mode()
