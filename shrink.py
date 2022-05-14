import numpy as np 
import pickle
import pandas as pd 
from collections import Counter
df=pd.read_csv('edited.csv')

N=df.userId.max()+1
M=df.movie_idx.max()+1
user_id_count=Counter(df.userId)
movie_id_count=Counter(df.movie_idx)
n=10000
m=2000
user_ids=[u for u,c in user_id_count.most_common(n)]
movie_ids=[m for m,c in movie_id_count.most_common(m)]
df_small=df[df.userId.isin(user_ids) & df.movie_idx.isin(movie_ids)].copy()
new_user_id_map={}
count=0
for old in user_ids:
    new_user_id_map[old]=count
    count+=1
new_movie_id_map={}
count=0
for old in movie_ids:
    new_movie_id_map[old]=count
    count+=1
df_small.loc[:,'userId']=df_small.apply(lambda row: new_user_id_map[row.userId],axis=1)
df_small.loc[:,'movie_idx']=df_small.apply(lambda row: new_movie_id_map[row.movie_idx],axis=1)

print("Size of Dataset",len(df_small))
df_small.to_csv('shrinked.csv') 