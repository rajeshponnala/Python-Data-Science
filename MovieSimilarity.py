import pandas as pd
import numpy as np


u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user',sep='|', names=u_cols)
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols)
m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(5),encoding='latin-1')
movies_ratings = pd.merge(movies,ratings)
lens = pd.merge(movies_ratings,users)


def getgenderPreference(g):
    s = len(g)
    mrper = (g['rating'][g['sex']=='M'].count()/s)*100
    frper = (g['rating'][g['sex']=='F'].count()/s)*100
    if mrper == frper:
        return 'N'
    elif mrper > frper:
        return 'M'
    else:
        return 'F'

def get_TransformedData(data):
    l = len(data['movie_id'].unique())
    z = sorted(data["movie_id"].unique())
    df4= pd.DataFrame(np.zeros((l,4)),index=z,columns=['release_date','avg_rating','gender_preference','age'])
    df4.index.name = 'Movie Id'
    df4['release_date'] = pd.cut(data["release_date"].map(lambda x: None if pd.isnull(x) else int(str(x).split('-')[-1])),bins=7,labels=['1922-1932','1933-1943','1944-1954','1955-1965','1966-1976','1977-1987','1988-1998'])
    df4['avg_rating'] = pd.cut(data.groupby('movie_id')['rating'].agg(np.mean),bins=5,labels=['R1','R2','R3','R4','R5'])
    df4['age']=pd.cut(data.groupby('movie_id')['age'].agg(np.mean),bins=[7,17,27,37,47,57,67,77],labels=['A1','A2','A3','A4','A5','A6','A7'])
    df4['gender_preference']= data.groupby('movie_id').apply(getgenderPreference)
    return df4

def getRank(movie1,movie2):
    rank = 0
    if movie1['release_date'] ==  movie2['release_date']:
        rank+=1
    if movie1['avg_rating'] ==  movie2['avg_rating']:
        rank+=1
    if movie1['gender_preference'] ==  movie2['gender_preference']:
        rank+=1
    if movie1['age'] ==  movie2['age']:
        rank+=1
    return rank

def getSimilarityTable(data):
    data = data.head(100)
    transformedData=get_TransformedData(data)
    l= len(data['movie_id'].unique())
    z = sorted(data["movie_id"].unique())
    df5=pd.DataFrame(np.zeros((l,l)),index=z,columns=z)
    for a in df5.columns:
         for b in df5.index:
             df5.ix[b][a] = getRank(transformedData.ix[b],transformedData.ix[a])
    return df5

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

#def userLikesOrDislikesMovie(user,movie):
 #   data = lens.head(100)
 #   tData = get_TransformedData(data)
 #   simMovies = [ m for m in tData.ix[movie].index.drop(movie) if tData.ix[movie][m] == 4]
