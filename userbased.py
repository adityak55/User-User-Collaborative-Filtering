import numpy as np 
import pickle
import pandas as pd 
from collections import Counter
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle
from datetime import datetime
from sortedcontainers import SortedList

with open('user2movie.json','rb') as f:
    user2movie=pickle.load(f)
    
    
with open('movie2user.json','rb') as f:
    movie2user=pickle.load(f)
    
with open('usermovie2rating.json','rb') as f:
    usermovie2rating=pickle.load(f)
    
with open('usermovie2rating_test.json','rb') as f:
    usermovie2rating_test=pickle.load(f)
    
N=np.max(list(user2movie.keys()))+1
m1=np.max(list(movie2user.keys()))+1
m2=np.max([m for (u,m),r in usermovie2rating_test.items()])
M=max(m1,m2)+1
if N>10000:
    print("okay")
    exit()
K=25
limit=5
neig=[]
averages=[]
deviat=[]

for i in range(N):
    movie_i=user2movie[i]
    movie_i_set=set(movie_i)
    rating_i={movie : usermovie2rating[(i,movie)] for movie in movie_i}
    avg_i=np.mean(list(rating_i.values()))
    dev_i={movie:(rating - avg_i) for movie,rating in rating_i.items()}
    dev_i_values=np.array(list(dev_i.values()))
    stigma_i=np.sqrt(dev_i_values.dot(dev_i_values))
    averages.append(avg_i)
    deviat.append(dev_i)
    
    
    sl=SortedList()
    for j in range(N):
        if j!=i:
            movie_j=user2movie[j]
            movie_j_set=set(movie_j)
            common_movie=(movie_i_set & movie_j_set)
            if len(common_movie)>limit:
                rating_j={movie : usermovie2rating[(j,movie)] for movie in movie_j}
                avg_j=np.mean(list(rating_j.values()))
                dev_j={movie:(rating - avg_j) for movie,rating in rating_j.items()}
                dev_j_values=np.array(list(dev_j.values()))
                stigma_j=np.sqrt(dev_j_values.dot(dev_j_values))
                num=sum(dev_i[m]*dev_j[m] for m in common_movie)
                wij=num/stigma_i*stigma_j
                sl.add((-wij,j))
                if len(sl)>K:
                    del sl[-1]
    neig.append(sl)
    if i%1==0:
        print(i)
def predict(i,m):
    num=0
    dem=0
    for wi_j,j in neig[i]:
        try:
            num+=-wi_j*deviat[j][m]
            dem+=abs(wi_j)
        except KeyError:
            pass
    if dem==0:
        predictions=averages[i]
    else:
        predictions=num/dem +averages[i]
    predictions=min(5,predictions)
    predictions=max(0.5,predictions)
    return predictions

train_predicts=[]
train_target=[]
for (i,m),target in usermovie2rating.items():
    predictions=predict(i,m)
    train_predicts.append(predictions)
    train_target.append(target)

test_predicts=[]
test_target=[]
for (i,m),target in usermovie2rating_test.items():
    predictions=predict(i,m)
    test_predicts.append(predictions)
    test_target.append(target)
def mse(p,t):
    p=np.array(p)
    t=np.array(t)
    return np.mean((p-t)**2)
print("train mse: ",mse(train_predicts,train_target))
print("test_mse: ",mse(test_predicts,test_target))

        
    
                
                
            
    
    
    

