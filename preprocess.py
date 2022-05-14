import pandas as pd 
df=pd.read_csv("F:\\reseach paper\\internreport\\rec\\rating.csv")
df.userId =df.userId-1
unique_movieid=set(df.movieId.values)
count=0
movieidx={}
for movie in unique_movieid:
    movieidx[movie]=count
    count+=1
df['movie_idx']=df.apply(lambda row :movieidx[row.movieId],axis=1)
df=df.drop(columns=['timestamp'])
df.to_csv('edited.csv')