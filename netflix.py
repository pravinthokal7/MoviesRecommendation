# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 14:32:16 2018

@author: pravin
"""
import pandas as pd
from tkinter import *

movies = pd.read_csv("final.csv", encoding='latin1');

#print(movies)

from sklearn.feature_extraction.text import TfidfVectorizer
 
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english') 
tfidf_matrix = tf.fit_transform(movies['genres'])


from sklearn.metrics.pairwise import linear_kernel 
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# Build a 1-dimensional array with movie titles 
titles = movies['title'] 
indices = pd.Series(movies.index, index=movies['title'])

# Function that get movie recommendations based on the cosine similarity score of movie genres 
def genre_recommendations(title): 
    idx = indices[title]    
    sim_scores = list(enumerate(cosine_sim[idx]))    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)    
    sim_scores = sim_scores[1:21]    
    movie_indices = [i[0] for i in sim_scores]    
    return titles.iloc[movie_indices]


def search():
    a = Textentry.get()
    op=genre_recommendations(a).head(20)
    output.insert(END,op)


#window
window = Tk()
window.title("Netflix")

#label1
Label (window, text="Enter the Movie" ) .grid(row=0,column=0)

#textbox
Textentry = Entry(window,width=60)
Textentry.grid(row=0,column=1)

#button
Button (window, text = "search",width=6,command=search). grid(row=0,column=2)

#label1
Label (window, text="Recommended List" ) .grid(row=2,column=1)

#output
output = Text(window,width=60,height=25)
output.grid(row=3,column=1)

#main
window.mainloop()
