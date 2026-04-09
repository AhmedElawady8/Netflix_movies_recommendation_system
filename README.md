# Netflix Movie Recommendation System

A content-based movie recommendation system that suggests similar movies using Machine Learning (Cosine Similarity) with a beautiful Netflix-themed Streamlit UI. The app fetches live movie data from the TMDB API and displays posters, ratings, genres, and plot overviews for each recommendation.

---

## 🚀 Live Demo

[👉 Open the App on Streamlit](YOUR_STREAMLIT_LINK_HERE)

---

## ✨ Features

- Recommends 5 similar movies based on your selection
- Fetches live movie posters from the TMDB API
- Displays rating, genre, runtime, and plot for each movie
- Netflix-style image carousel on the homepage
- Full dark theme matching Netflix's design
- Fast performance using cached model files (Pickle)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| Pandas and NumPy | Data processing and manipulation |
| Scikit-learn | CountVectorizer + Cosine Similarity |
| Streamlit | Web application framework |
| Pickle | Saving and loading the ML model |
| TMDB API | Fetching live movie posters and details |
| Svelte | Custom image carousel component |

---

## 📁 Project Structure
Netflix_movies_recommendation_system/
├── app.py                        # Main Streamlit app (user interface)
├── Main.py                       # Builds the ML model and similarity matrix
├── movies_list.pkl               # Saved cleaned movie DataFrame
├── similarity.pkl                # Cosine similarity matrix (model output)
├── requirements.txt              # Python dependencies
├── .gitignore                    # Files Git should ignore
├── Data/
│   └── top10K-TMDB-movies.csv   # Raw TMDB movie dataset (top 10K movies)
└── frontend/
├── public/                   # Compiled Svelte carousel component
├── src/                      # Carousel source code
├── package.json              # Node.js dependencies
├── rollup.config.js          # Svelte bundler config
└── tsconfig.json             # TypeScript config
---

## ⚙️ How It Works

1. **Data Loading** — Reads the top 10K TMDB movies dataset
2. **Feature Engineering** — Combines overview and genre into a tags column
3. **Vectorization** — Converts text tags into numeric vectors using CountVectorizer
4. **Similarity** — Computes cosine similarity between all movies
5. **Recommendation** — Given a movie, finds the top 5 most similar ones
6. **Live Data** — Fetches posters and details from TMDB API in real time

---

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/AhmedElawady8/Netflix_movies_recommendation_system.git
cd Netflix_movies_recommendation_system
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Generate the model files**
```bash
python Main.py
```

**4. Run the app**
```bash
streamlit run app.py
```

---

## 📊 Dataset

- **Source:** TMDB (The Movie Database)
- **Size:** Top 10,000 movies
- **Date:** Data up to July 26, 2022
- **Features used:** id, title, genre, overview

---

## 🔑 API Key

This project uses the TMDB API to fetch live movie posters, ratings, genres, and overviews.

[https://www.themoviedb.org/documentation/api](https://www.themoviedb.org/documentation/api)

---

## 📄 License

This project is licensed under the MIT License.
