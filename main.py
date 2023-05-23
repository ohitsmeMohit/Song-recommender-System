import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Data Collection
# Assume you have a dataset called 'songs.csv' with columns: user_id, song_id, song_name, artist_name, rating

# Step 2: Data Preprocessing
data = pd.read_csv('songs.csv')
train_data, test_data = train_test_split(data, test_size=0.2)  # Split the data into training and testing sets

# Step 3: Feature Representation
user_song_matrix = train_data.pivot(index='user_id', columns='song_id', values='rating').fillna(0)

# Step 4: Model Training
# Calculate the cosine similarity between users
user_similarity = cosine_similarity(user_song_matrix)

# Step 5: Model Evaluation
# Predict the ratings for the test set
test_data['predicted_rating'] = [user_song_matrix.loc[user_id, song_id] for user_id, song_id in zip(test_data['user_id'], test_data['song_id'])]

# Step 6: Model Deployment
# You can use the trained model to make recommendations based on a user's preferences

# Create TF-IDF vectorizer for song names and artist names
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the song name and artist name columns to obtain TF-IDF features
song_name_tfidf = tfidf_vectorizer.fit_transform(data['song_name'].values.astype('U'))
artist_name_tfidf = tfidf_vectorizer.fit_transform(data['artist_name'].values.astype('U'))

# Example recommendation function
def recommend_songs(user_id, query, top_n=5):
    user_ratings = user_song_matrix.loc[user_id]
    similar_users = user_similarity[user_id].argsort()[::-1][1:]  # Get similar users based on cosine similarity

    # Calculate the average rating of each unrated song among similar users
    song_ratings = {}
    for song_id, rating in user_ratings.iteritems():
        if rating == 0:
            song_name = data[data['song_id'] == song_id]['song_name'].values[0]
            artist_name = data[data['song_id'] == song_id]['artist_name'].values[0]
            song_name_similarity = cosine_similarity(tfidf_vectorizer.transform([query]), tfidf_vectorizer.transform([song_name]))
            artist_name_similarity = cosine_similarity(tfidf_vectorizer.transform([query]), tfidf_vectorizer.transform([artist_name]))
            song_ratings[song_id] = (song_name_similarity + artist_name_similarity) / 2

    # Sort the songs based on average ratings and recommend the top N songs
    recommended_songs = sorted(song_ratings, key=song_ratings.get, reverse=True)[:top_n]
    return recommended_songs

# Example usage
user_id = 1
query = "Ed Sheeran"
recommended_songs = recommend_songs(user_id, query, top_n=5)
print("Recommended songs for user", user_id, "based on query:", query)
for song_id in recommended_songs:
    song_name = data[data['song_id'] == song_id]['song_name'].values[0]
    print(song_name)

    
# Recommended songs for user 1 based on query: Ed Sheeran
# "Someone Like You"
# "Hotel California"
