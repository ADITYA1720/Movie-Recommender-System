import streamlit as st
import pandas as pd

df = pd.read_csv('ratings.csv')

def login():
    st.title("Login Page")

    # Create input widgets for username and password
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        # Perform authentication
        if authenticate(username):
            st.success("Login successful!")
        else:
            st.error("Invalid username or password. Please try again.")

def authenticate(username):
    if int(username) in df['userId'].values:
        # For simplicity, we're not checking passwords in this example
        return True
    else:
        return False
    

if __name__ == "__main__":
    login()











# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# ratings = pd.read_csv('ratings.csv')
# movies = pd.read_csv('movies.csv')

# ## Overview of the data

# n_ratings = len(ratings)
# n_movies = ratings['movieId'].nunique()
# n_users = ratings['userId'].nunique()

# print(f"Number of ratings: {n_ratings}")
# print(f"Number of unique movieId's: {n_movies}")
# print(f"Number of unique users: {n_users}")
# print(f"Average number of ratings per user: {round(n_ratings/n_users, 2)}")
# print(f"Average number of ratings per movie: {round(n_ratings/n_movies, 2)}")

# ## Visualize the data
# sns.countplot(x="rating", data=ratings, palette="viridis")
# plt.title("Distribution of movie ratings", fontsize=14)
# plt.show()

# ## List the Number of Different Genres with their occurence frequency
# movies['genres'] = movies['genres'].apply(lambda x: x.split("|"))
# from collections import Counter
# genre_frequency = Counter(g for genres in movies['genres'] for g in genres)
# print(f"There are {len(genre_frequency)} genres with {genre_frequency}")
# print("The 5 most common genres: \n", genre_frequency.most_common(5))

# ## Feature Enhancement and data tranformations
