
from django.shortcuts import render

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django.conf import settings

import pandas as pd
import numpy as np 
import ast


movies = pd.read_csv(f'{settings.BASE_DIR}/app/data/movies.csv')
all_available_movies = list(movies['title'])
credits = pd.read_csv(f'{settings.BASE_DIR}/app/data/credit.csv')

movies = movies.merge(credits,on='title')
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

def filter_dataset(text):
    data = []
    for i in ast.literal_eval(text):
        data.append(i['name']) 
    return data

def filter_dataset1(text):
    data = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            data.append(i['name'])
        counter+=1
    return data 

def director_name(text):
    director = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            director.append(i['name'])
    return director 

def delete(item):
    item1 = []
    for i in item:
        item1.append(i.replace(" ",""))
    return item1

movies.dropna(inplace=True)
movies['keywords'] = movies['keywords'].apply(filter_dataset)
movies['keywords'] = movies['keywords'].apply(delete)
movies['crew'] = movies['crew'].apply(director_name)
movies['crew'] = movies['crew'].apply(delete)
movies['genres'] = movies['genres'].apply(filter_dataset)
movies['genres'] = movies['genres'].apply(delete)
movies['cast'] = movies['cast'].apply(filter_dataset)
movies['cast'] = movies['cast'].apply(lambda x:x[0:3])
movies['cast'] = movies['cast'].apply(delete)

movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
new['tags'] = new['tags'].apply(lambda x: " ".join(x))


cv = CountVectorizer(max_features=5000,stop_words='english')
vector = cv.fit_transform(new['tags']).toarray()


similarity = cosine_similarity(vector)
new[new['title'] == 'The Lego Movie'].index[0]


def suggest(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    movies = []
    for i in distances[1:6]:
        movies.append(new.iloc[i[0]].title)
    return movies
    


def home_page(request):
    return render(request, 'app/home.html')


def recommend_movie(request):
    context = {}
    if request.method == 'POST':
        movie_name = request.POST.get('movie_name')
        if movie_name in all_available_movies:
            movies = suggest(movie_name)
            context['available'] = True
            context['movies'] = movies
        else:
            context['available'] = False 
            context['show_error'] = True
    return render(request, 'app/home.html', context)