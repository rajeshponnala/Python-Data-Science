import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user',sep='|', names=u_cols)

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols)

m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(5),encoding='latin-1')

movies_ratings = pd.merge(movies,ratings)
lens = pd.merge(movies_ratings,users)

#most_rated = lens.groupby('title').size().order(ascending=False)[:25]
most_rated = lens.title.value_counts()
#print(most_rated[:25])

movie_stats = lens.groupby('title').agg({'rating':[np.size,np.mean]})
#print(highly_rated_movies.sort([('rating','mean')],ascending=False).head())

atleast_100 = movie_stats['rating']['size']>=100
#print(movie_stats[atleast_100].sort([('rating', 'mean')], ascending=False)[:15])

users.age.hist(bins=30)
plt.title("Distribution of users' ages")
plt.ylabel('count of users')
plt.xlabel('age')
#plt.show()

labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
lens['age_group'] = pd.cut(lens.age, range(0, 81, 10), right=False, labels=labels)
#print(lens.groupby('age_group').agg({'rating': [np.size, np.mean]}))

most_50 = lens.groupby('movie_id').size().order(ascending=False)[:50]

lens.set_index('movie_id', inplace=True)

by_age = lens.ix[most_50.index].groupby(['title', 'age_group'])
by_age.rating.mean().unstack(1).fillna(0)[10:20]

lens.reset_index('movie_id', inplace=True)
