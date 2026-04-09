import streamlit as st
import streamlit.components.v1 as components
import pickle
import requests
import os

# ============================================================
# STEP 1: PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title=" Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

# ============================================================
# STEP 2: CUSTOM DARK THEME STYLING (Netflix look)
# ============================================================
st.markdown("""
    <style>
        .stApp { background-color: #141414; color: white; }

        h1 { color: #E50914; font-family: 'Arial Black', sans-serif; font-size: 2rem; }

        .stMarkdown, .stText, label, .stSelectbox label { color: white !important; }

        .stSelectbox > div > div {
            background-color: #333333 !important;
            color: white !important;
            border: 1px solid #555 !important;
            border-radius: 4px;
        }

        /* Main recommend button */
        .stButton > button {
            background-color: transparent;
            color: #E50914;
            border: 2px solid #E50914;
            border-radius: 4px;
            padding: 0.5rem 1.5rem;
            font-size: 1rem;
            font-weight: bold;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        .stButton > button:hover { background-color: #E50914; color: white; }

        /* Movie title text under poster */
        .movie-title {
            color: white;
            font-size: 0.9rem;
            font-weight: bold;
            margin-top: 6px;
            text-align: center;
        }

        /* "More Info" expander — dark card look */
        .streamlit-expanderHeader {
            background-color: #1f1f1f !important;
            color: #E50914 !important;
            border-radius: 6px !important;
            font-size: 0.8rem !important;
        }
        .streamlit-expanderContent {
            background-color: #1f1f1f !important;
            color: #cccccc !important;
            border-radius: 0 0 6px 6px !important;
            font-size: 0.82rem;
        }

        /* Rating badge */
        .rating-badge {
            display: inline-block;
            background-color: #E50914;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.78rem;
            font-weight: bold;
            margin-bottom: 6px;
        }

        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    </style>
""", unsafe_allow_html=True)


# ============================================================
# STEP 3: API HELPER FUNCTIONS
# ============================================================

TMDB_API_KEY = "c7ec19ffdd3279641fb606d19ceb9bb1"

def fetch_poster(movie_id):
    """Returns the full poster image URL for a given TMDB movie ID."""
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        data = requests.get(url, timeout=5).json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
    except Exception:
        pass
    return "https://via.placeholder.com/500x750.png?text=No+Image"


def fetch_movie_details(movie_id):
    """
    Fetches full movie details from TMDB API.
    Returns a dict with: overview, rating, release_year, genres, runtime.
    """
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        data = requests.get(url, timeout=5).json()

        overview  = data.get('overview', 'No description available.')
        rating    = round(data.get('vote_average', 0), 1)
        year      = data.get('release_date', 'N/A')[:4]
        genres    = ", ".join([g['name'] for g in data.get('genres', [])])
        runtime   = data.get('runtime', 0)
        runtime_str = f"{runtime} min" if runtime else "N/A"

        return {
            "overview": overview,
            "rating":   rating,
            "year":     year,
            "genres":   genres,
            "runtime":  runtime_str
        }
    except Exception:
        return {
            "overview": "Could not load details.",
            "rating": "N/A", "year": "N/A",
            "genres": "N/A", "runtime": "N/A"
        }


# ============================================================
# STEP 4: LOAD SAVED MODEL FILES (cached for performance)
# ============================================================
@st.cache_resource
def load_model():
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Build movies_list.pkl from CSV if not exists
    if not os.path.exists("movies_list.pkl"):
        df = pd.read_csv("Data/top10K-TMDB-movies.csv")
        df['tags'] = df['overview'].fillna('') + ' ' + df['genre'].fillna('')
        movies = df[['id', 'title', 'tags']]
        pickle.dump(movies, open("movies_list.pkl", "wb"))
    else:
        movies = pickle.load(open("movies_list.pkl", "rb"))

    # Build similarity.pkl from movies if not exists
    if not os.path.exists("similarity.pkl"):
        cv = CountVectorizer(max_features=10000, stop_words='english')
        vector = cv.fit_transform(movies['tags'].values.astype('U')).toarray()
        similarity = cosine_similarity(vector)
        pickle.dump(similarity, open("similarity.pkl", "wb"))
    else:
        similarity = pickle.load(open("similarity.pkl", "rb"))

    return movies, similarity

movies, similarity = load_model()
movies_list = movies['title'].values
# ============================================================
# STEP 5: APP HEADER
# ============================================================
st.header("Netflix Movie Recommender System 🎬")
st.caption(" Movies till 26-July-2022✨")


# ============================================================
# STEP 6: IMAGE CAROUSEL (Custom Svelte Component)
# ============================================================

# FIX: Create empty bundle.css if missing to silence FileNotFoundError
css_path = os.path.join("frontend", "public", "build", "bundle.css")
if not os.path.exists(css_path):
    os.makedirs(os.path.dirname(css_path), exist_ok=True)
    with open(css_path, "w") as f:
        f.write("")

imageCarouselComponent = components.declare_component(
    "image-carousel-component",
    path="frontend/public"
)

carousel_movie_ids = [
    1632, 299536, 17455, 2830, 429422,
    9722, 13972, 240, 155, 598,
    914, 255709, 572154
]
carousel_urls = [fetch_poster(mid) for mid in carousel_movie_ids]
imageCarouselComponent(imageUrls=carousel_urls, height=200)


# ============================================================
# STEP 7: MOVIE SELECTION DROPDOWN
# ============================================================
st.markdown("**Type or select a movie from the dropdown**")
selected_movie = st.selectbox(
    label="Select a movie",
    label_visibility="collapsed",
    options=movies_list
)


# ============================================================
# STEP 8: RECOMMENDATION FUNCTION
# Returns top 5 similar movies with their IDs, names, posters
# ============================================================
def recommend(movie_title):
    """Returns top 5 recommended movies: (names, posters, ids)"""
    index = movies[movies['title'] == movie_title].index[0]

    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )

    names   = []
    posters = []
    ids     = []

    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].id
        ids.append(movie_id)
        names.append(movies.iloc[i[0]].title)
        posters.append(fetch_poster(movie_id))

    return names, posters, ids


# ============================================================
# STEP 9: SHOW RECOMMENDATIONS + "MORE INFO" EXPANDER
# Each card: poster → title → expandable plot/rating/genre
# ============================================================
if st.button("Show Recommendation"):

    movie_names, movie_posters, movie_ids = recommend(selected_movie)

    cols = st.columns(5)

    for col, name, poster, mid in zip(cols, movie_names, movie_posters, movie_ids):
        with col:
            # Movie title above poster
            st.markdown(f'<div class="movie-title">{name}</div>', unsafe_allow_html=True)

            # Poster image
            st.image(poster, use_column_width=True)

            # Expandable "More Info" section — click to reveal details
            with st.expander("ℹ️ More Info"):
                details = fetch_movie_details(mid)

                # Star rating badge
                st.markdown(
                    f'<span class="rating-badge">⭐ {details["rating"]} / 10</span>',
                    unsafe_allow_html=True
                )

                # Year and runtime on same line
                st.markdown(
                    f"📅 **{details['year']}** &nbsp;|&nbsp; 🕐 **{details['runtime']}**",
                    unsafe_allow_html=True
                )

                # Genre tags
                if details['genres']:
                    st.markdown(f"🎭 _{details['genres']}_")

                st.markdown("---")

                # Plot overview (the main description)
                st.markdown(details['overview'])