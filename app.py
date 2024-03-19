import streamlit as st
import pickle
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

#Set page title and icon
st.set_page_config(
    page_title="Movie Recommender App",
    page_icon="🎬"
)

#import dataframes
merge_movie =  pickle.load(open('merge_movie_info.pkl', 'rb'))
movies =  pickle.load(open('movies.pkl', 'rb'))
user_similarity =  pickle.load(open('user_similarity.pkl', 'rb'))
user_item_matrix =  pickle.load(open('user_item_matrix.pkl', 'rb'))
genre_list =  pickle.load(open('genres.pkl', 'rb'))
movie_genres = genre_list['genres'].values
title_list = movies['title'].values

#calculating cosine similarity

# Preprocess genres (convert to lowercase and remove spaces)
movies['genres'] = movies['genres'].str.lower().str.replace(' ', '')

# Create a TF-IDF vectorizer to convert genres into numerical features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['genres'])
# Calculate cosine similarity between movies based on genres
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
similarity = cosine_sim



#side bar
def side_bar():
    st.sidebar.title("Movie Recommender Overview")
    st.sidebar.write("Welcome to the Movie Recommender App! This app provides recommendations based on different algorithms.")
    st.sidebar.write("This is part of a certification project submitted by Kingsley Amemayor.")

    st.sidebar.subheader("List of recommendation algorithms")
    recommendation_type = st.sidebar.selectbox("Choose Recommendation Type", ["Popularity-Based Genre", "Content-Based", "Collaborative-Based"])
    return recommendation_type

# Popularity-Based Recommender System at Genre Level

def popularity_based_genre_recommendation(genre, rating, recommendation):
    genre_filtered = merge_movie[merge_movie['genres'].str.contains(genre, case=False, na=False)]
    idx_max_rating = genre_filtered.groupby('title')['rating'].idxmax()
    filtered_df = genre_filtered.loc[idx_max_rating]
    high_rated_filtered = filtered_df[filtered_df['rating'] >= rating]
    top_N_recommendations = high_rated_filtered.sort_values(by=['rating'], ascending=False).head(recommendation)

    return top_N_recommendations[['title', 'rating', 'num_reviews']]

# Content-Based Recommender System
def content_based_recommendation(movie_title, N):
    idx = movies.index[movies['title'] == movie_title].tolist()[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:N+1]  # Exclude the movie itself (index 0) and get top N
    movie_indices = [i[0] for i in sim_scores]

    return movies['title'].iloc[movie_indices]
    #return movies['title'].iloc[movie_indices]

# Collaborative-Based Recommender System
def collaborative_based_recommendation(target_user, n_recommendations, k_similar_users):
        # Find K most similar users to the target user
    similar_users_indices = user_similarity[target_user-1].argsort()[::-1][1:k_similar_users+1]  # Excluding the user itself
    
    # Calculate average rating of each movie among similar users
    avg_rating = user_item_matrix.iloc[similar_users_indices].mean(axis=0)
    
    # Filter out movies already rated by the target user
    user_movies = user_item_matrix.loc[target_user]
    avg_ratings = avg_rating[user_movies != 0]     #Since 0 is used to indicate no rating
    
    # Get top N movie recommendations
    top_movies = avg_ratings.sort_values(ascending=False).head(n_recommendations)
    
    # Retrieve movie titles from movieId
    top_movies_titles = merge_movie[merge_movie['movieId'].isin(top_movies.index)]['title']
    top_movies_titles = pd.DataFrame(top_movies_titles)
    top_movies_titles = top_movies_titles.drop_duplicates() #droping duplicated movie titles
    return top_movies_titles


# Streamlit App
def main():
    
    #side_bar()
    st.title("Edureka Certification Project")

    # User's choice
    recommendation_type = side_bar()

    # User input fields
    if recommendation_type == "Popularity-Based Genre":
        st.subheader("Popularity Based Recommendation")
        selected_genre = st.selectbox("Select Genre From Dropdown", movie_genres)
        value_of_rating = st.slider("Select minimum rating threshold (t):", 1.0, 5.0, 3.0, step=0.1)
        number_of_recommendations = st.slider("Select number of recommendations (N):", 1, 10, 5)

        if st.button("Show Recommendations"):
            recommendations = popularity_based_genre_recommendation(selected_genre, value_of_rating, number_of_recommendations)
            st.subheader("Popularity-Based Genre Recommendations")
            st.dataframe(recommendations, use_container_width=True)
    
    elif recommendation_type == "Content-Based":
        st.subheader("Content Based Recommendation")
        selected_title = st.selectbox("Select target movie title:", title_list)
        n_recommendations = st.slider("Select number of recommendations (N):", 1, 10, 5)

        if st.button("Show Recommendations"):
            recommendations = content_based_recommendation(selected_title, n_recommendations)
            st.subheader("Content-Based Recommendations")
            st.dataframe(recommendations, use_container_width=True)

    elif recommendation_type == "Collaborative-Based":
        st.subheader("Collaborative Based Recommendation")
        target_user = st.text_input("Enter target user:")
        
        if target_user.strip():  # Check if the input is not empty
            target_user = int(target_user)  # Convert to integer
        else:
            target_user = 1  # Set to None or any default value you prefer
        
        k_similar_users = st.slider("Select number of similar users (K):", 1, 100, 3)
        n_recommendations = st.slider("Select number of recommendations (N):", 1, 10, 5)

        if st.button("Show Recommendations"):
            recommendations = collaborative_based_recommendation(target_user, k_similar_users, n_recommendations)
            st.subheader("Collaborative-Based Recommendations")
            st.dataframe(recommendations, use_container_width=True)

# Run the app
if __name__ == "__main__":
    main()
