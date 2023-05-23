# Song-recommender-System

This project focuses on building a song recommendation system using collaborative filtering and content-based approaches. The project begins with data collection, assuming the existence of a dataset called 'songs.csv' containing information such as user IDs, song IDs, song names, artist names, and ratings. 

After data preprocessing, which involves splitting the dataset into training and testing sets, the program moves on to feature representation. It constructs a user-song matrix from the training data, where each cell represents a user's rating for a specific song. Any missing values in the matrix are filled with zeros.

The next step involves model training, specifically calculating the cosine similarity between users using the user-song matrix. This similarity matrix will be crucial for finding similar users during the recommendation process.

Model evaluation is then performed by predicting the ratings for the test set. The program retrieves user-song pairs from the test data and uses the user-song matrix to determine their predicted ratings.

For model deployment, the trained model can be utilized to make song recommendations based on a user's preferences. Additionally, the program incorporates a TF-IDF vectorizer to transform song names and artist names into TF-IDF features, enhancing the recommendation process.

The project includes an example recommendation function, 'recommend_songs', which takes a user ID and a query (representing a song or artist name) as inputs. It calculates the average ratings of unrated songs among similar users based on cosine similarity and recommends the top N songs. 

Overall, this project presents a comprehensive approach to song recommendation, combining collaborative filtering, content-based filtering, and similarity calculations to deliver personalized recommendations to users based on their preferences and the features of songs and artists.
