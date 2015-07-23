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

most_50 = lens.groupby('movie_id').size().order(ascending=False)[:50]

#most_rated = lens.groupby('title').size().order(ascending=False)[:25]

def getmostrated(data):
    return data.title.value_counts()

def get_highlyrated_moviesatleast100(data):
    movie_stats = lens.groupby('title').agg({'rating':[np.size,np.mean]})
    return  movie_stats[ movie_stats['rating']['size']>=100].sort([('rating', 'mean')], ascending=False)

def plot_users_age_distribution_for_movies(users):
    users.age.hist(bins=30)
    plt.title("Distribution of users' ages")
    plt.ylabel('count of users')
    plt.xlabel('age')
    plt.show()

def get__50_movies_by_age_group(data):
    labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
    data['age_group'] = pd.cut(data.age, range(0, 81, 10), right=False, labels=labels)
    data.set_index('movie_id', inplace=True)
    by_age = data.ix[most_50.index].groupby(['title', 'age_group'])
    return  by_age.rating.mean().unstack(1).fillna(0)[10:20]

def plot_diff_between_male_female_rating(data):
    data.reset_index('movie_id', inplace=True)
    pivoted = data.pivot_table(index=['movie_id', 'title'],
                           columns=['sex'],
                           values='rating',
                           fill_value=0)
    pivoted['diff']=pivoted.M - pivoted.F
    pivoted.reset_index('movie_id', inplace=True)
    disagreements = pivoted[pivoted.movie_id.isin(most_50)]["diff"]
    disagreements.order().plot(kind='barh',figsize=[9, 15])
    plt.title('Male vs. Female Avg. Ratings\n(Difference > 0 = Favored by Men)')
    plt.ylabel('Title')
    plt.xlabel('Average Rating Difference')
    plt.show()

def pearson(rating1, rating2):
    commonr=pd.concat([rating1, rating2]).dropna().index.get_duplicates()
    n = len(commonr)
    if n == 0 :
        return None
    denominator = math.sqrt((rating1.ix[commonr]**2).sum()-((rating1.ix[commonr].sum()**2)/n)) * math.sqrt((rating2.ix[commonr]**2).sum()-((rating2.ix[commonr].sum()**2)/n))
    return 0 if denominator == 0 else ((rating1.ix[commonr]*rating2.ix[commonr]).sum()-(rating1.ix[commonr].sum()*rating2.ix[commonr].sum()/n))/denominator

def manhattan(rating1,rating2):
    return sum(abs((rating1-rating2).dropna()))

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
