
# ============================================================
# NETFLIX MOVIE RECOMMENDATION SYSTEM
# Uses: Pandas, Scikit-learn, Cosine Similarity, Pickle
# ============================================================

# --- Import required libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# STEP 1: LOAD THE DATA
# Read the CSV file into a pandas DataFrame
# ============================================================
movies_df = pd.read_csv("../Netfilx_Recommendatio/Data/top10K-TMDB-movies.csv")


# ============================================================
# STEP 2: EXPLORE THE DATA
# Get a quick look at what the data looks like
# ============================================================

# Show the first 5 rows
movies_df.head()

# Show column names, data types, and non-null counts
movies_df.info()

# Show basic statistics (count, mean, min, max, etc.)
movies_df.describe()

# Show the first 50 unique movie titles
movies_df["title"][:50].unique()

# Count how many times each title appears
movies_df["title"].value_counts()

# Show all unique genres in the dataset
movies_df["genre"].unique()

# Count how many movies belong to each genre
movies_df["genre"].value_counts()

# Show only the title and popularity columns
movies_df[["title", "popularity"]]


# ============================================================
# STEP 3: CHECK DATA QUALITY
# Look for missing values and duplicate rows
# ============================================================

# Count missing (NaN) values in each column
print("Missing values:\n", movies_df.isna().sum())

# Count duplicate rows
print("Duplicate rows:", movies_df.duplicated().sum())

# Show all column names
print("Columns:", movies_df.columns.tolist())


# ============================================================
# STEP 4: SELECT ONLY THE COLUMNS WE NEED
# Keep: id, title, genre, overview
# Drop everything else to keep the model simple
# ============================================================
movies_df = movies_df[['id', 'title', 'genre', 'overview']]


# ============================================================
# STEP 5: CREATE THE "TAGS" FEATURE
# Combine 'overview' and 'genre' into one text column called 'tags'
# This gives the model richer text to compare movies by
# FIX: Added a space " " between overview and genre to avoid merging words
# FIX: Used fillna('') to handle any missing values safely
# ============================================================
movies_df["tags"] = (
    movies_df["overview"].fillna('') + " " + movies_df["genre"].fillna('')
)


# ============================================================
# STEP 6: BUILD THE FINAL CLEAN DATASET
# Drop the original 'genre' and 'overview' columns
# We no longer need them — 'tags' has their combined content
# ============================================================
clean_data = movies_df.drop(columns=['genre', 'overview']).reset_index(drop=True)


# ============================================================
# STEP 7: VECTORIZE THE TEXT (Convert words → numbers)
# CountVectorizer counts how often each word appears in 'tags'
# max_features=10000 → only use the 10,000 most common words
# stop_words='english' → ignore common words like "the", "a", "is"
# ============================================================
cv = CountVectorizer(max_features=10000, stop_words='english')

# Fit and transform the tags column into a numeric matrix
# .values.astype('U') converts to unicode string safely
vector = cv.fit_transform(clean_data['tags'].values.astype('U')).toarray()

print("Vector shape:", vector.shape)  # (num_movies, num_words)


# ============================================================
# STEP 8: COMPUTE COSINE SIMILARITY
# Compare every movie to every other movie
# Result is an N×N matrix where 1 = identical, 0 = no overlap
# ============================================================
similarity = cosine_similarity(vector)
print("Similarity matrix shape:", similarity.shape)


# ============================================================
# STEP 9: BUILD THE RECOMMENDATION FUNCTION
# Given a movie title, find the 10 most similar movies
# FIX: Renamed parameter from 'movies' to 'movie_title' to avoid
#      confusion with the DataFrame variable
# IMPROVEMENT: Returns a list instead of only printing
# ============================================================
def recommend(movie_title):
    """
    Given a movie title, return the top 10 most similar movies.

    Args:
        movie_title (str): The exact title of a movie in the dataset.

    Returns:
        list: Top 10 recommended movie titles (excluding the input movie).
    """
    # Find the row index of the given movie
    matches = clean_data[clean_data['title'] == movie_title]

    # Handle the case where the movie is not found
    if matches.empty:
        print(f"Movie '{movie_title}' not found in the dataset.")
        return []

    index = matches.index[0]

    # Sort all movies by similarity score (highest first)
    # enumerate() adds the index so we know which movie each score belongs to
    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )

    # Collect the top 10 results (skip index 0 — that's the movie itself)
    recommendations = []
    for i in distances[1:11]:
        recommendations.append(clean_data.iloc[i[0]].title)

    return recommendations


# ============================================================
# STEP 10: TEST THE FUNCTION
# Try recommending movies similar to "The Godfather"
# ============================================================
results = recommend("The Godfather")
print("\nMovies similar to 'The Godfather':")
for i, title in enumerate(results, 1):
    print(f"  {i}. {title}")


# ============================================================
# STEP 11: SAVE THE MODEL WITH PICKLE
# Save the cleaned movie list and the similarity matrix
# so we can load them later in the Streamlit app
# without re-running all the heavy computation
# ============================================================

# Save the cleaned movie DataFrame
pickle.dump(clean_data, open('movies_list.pkl', 'wb'))

# Save the similarity matrix
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("\nModel saved successfully!")
print("Files created: movies_list.pkl, similarity.pkl")