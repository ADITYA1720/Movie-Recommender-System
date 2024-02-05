import pandas as pd
import seaborn as sns
import streamlit as st
from collections import Counter

# Load data
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# Overview of the data
n_ratings = len(ratings)
n_movies = ratings['movieId'].nunique()
n_users = ratings['userId'].nunique()

# Display data overview
st.write(f"Number of ratings: {n_ratings}")
st.write(f"Number of unique movie Ids: {n_movies}")
st.write(f"Number of unique users: {n_users}")
st.write(f"Average number of ratings per user: {round(n_ratings/n_users, 2)}")
st.write(f"Average number of ratings per movie: {round(n_ratings/n_movies, 2)}")

# Visualize the data
st.set_option('deprecation.showPyplotGlobalUse', False)  # To prevent warnings
fig, ax = plt.subplots()
sns.countplot(x="rating", data=ratings, palette="viridis", ax=ax)
ax.set_title("Distribution of movie ratings")
st.pyplot(fig)

# List the Number of Different Genres with their occurrence frequency
movies['genres'] = movies['genres'].apply(lambda x: x.split("|"))
genre_frequency = Counter(g for genres in movies['genres'] for g in genres)

# Display genre information
st.write(f"There are {len(genre_frequency)} genres with:")
for genre, frequency in genre_frequency.items():
    st.write(f"{genre}: {frequency}")
st.write("The 5 most common genres:")
st.write(genre_frequency.most_common(5))
