import pandas as pd
from flask import Flask, render_template, request
from keras.models import load_model
import requests
from imdb import IMDb
from bs4 import BeautifulSoup
from preprocessing import preprocess
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer as wnl
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging
import random

app = Flask(__name__)

imdb_reviews = []

# Load the sentiment analysis model
model = load_model('sentiment_analysis_model.h5')

def analyze_movie_sentiment(movie_title):
    global imdb_reviews  # Use the global variable
    print(f"Fetching reviews for movie: {movie_title}")
    
    def search_movie_by_title(movie_title):
        ia = IMDb()
        search_result = ia.search_movie(movie_title)
        if search_result:
            return search_result[0].getID()
        return None

    def get_full_movie_info(movie_id):
        ia = IMDb()
        movie = ia.get_movie(movie_id)
        ia.update(movie, info=['reviews'], reviews_max_depth=100)
        return movie

    # Fetch IMDb reviews
    try:
        ia = IMDb()
        search_result = ia.search_movie(movie_title)
        if search_result:
            movie_id = search_result[0].getID()
            movie = ia.get_movie(movie_id)
            ia.update(movie, info=['reviews'])
            imdb_reviews = [review['content'] for review in movie['reviews']]
            print(f"Fetched {len(imdb_reviews)} IMDb reviews")
        else:
            imdb_reviews = []
            print("Movie not found on IMDb")
    except Exception as e:
        imdb_reviews = []
        print(f"Error fetching IMDb reviews: {e}")
        logging.error(f"Error fetching IMDb reviews: {e}")

    def get_metacritic_reviews(movie_title, max_pages=2):
        # Replace spaces with '-' and convert to lowercase
        formatted_movie_title = movie_title.replace(" ", "-").lower()

        # Access the movie's main page directly
        url = f"https://www.metacritic.com/movie/{formatted_movie_title}"

        # Use a custom User-Agent header to mimic a real browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
        }

        # Create a session and set the maximum number of redirects
        session = requests.Session()
        session.max_redirects = 10

        response = session.get(url, headers=headers)

        if response.status_code == 200:
            content = response.content
            soup = BeautifulSoup(content, "html.parser")
        else:
            print("Movie not found")

        meta_reviews = []

        for page in range(0, max_pages):
            # Get critic reviews
            url = f"https://www.metacritic.com/movie/{formatted_movie_title}/critic-reviews?page={page}"
            response = session.get(url, headers=headers).content

            soup = BeautifulSoup(response, "html.parser")

            for review in soup.find_all("div", class_="summary"):
                meta_reviews.append(review.find("a", class_="no_hover").text.strip())

            # Get user reviews
            url = f"https://www.metacritic.com/movie/{formatted_movie_title}/user-reviews?page={page}"
            response = session.get(url, headers=headers).content

            soup = BeautifulSoup(response, "html.parser")

            for review in soup.find_all("div", class_="review_body"):
                meta_reviews.append(review.text.strip())

        data = {"review": meta_reviews}  # Include both critic and user reviews
        metacritic_reviews_df = pd.DataFrame(data)
        return metacritic_reviews_df
    
    data = {"review": imdb_reviews}
    imdb_reviews_df = pd.DataFrame(data)

    # Fetch Metacritic reviews
    metacritic_reviews_df = get_metacritic_reviews(movie_title)
    print(f"Fetched {len(metacritic_reviews_df)} Metacritic reviews")

    # Check if the dataframes are not empty before running the code
    if not imdb_reviews_df.empty:
        # Preprocess the IMDb reviews
        imdb_reviews_preprocessed = preprocess(imdb_reviews_df)
        # Use the model to predict sentiment scores
        imdb_probabilities = model.predict(imdb_reviews_preprocessed)
        # Count the number of positive predictions
        imdb_positive_count = (imdb_probabilities).sum()
        # Calculate the sentiment score
        imdb_sentiment_score = imdb_positive_count / len(imdb_probabilities)
    else:
        # Return 0 as the score if the dataframe is empty
        imdb_sentiment_score = 0

    if not metacritic_reviews_df.empty:
        # Preprocess the Metacritic reviews
        metacritic_reviews_preprocessed = preprocess(metacritic_reviews_df)
        # Use the model to predict sentiment scores
        metacritic_probabilities = model.predict(metacritic_reviews_preprocessed)
        # Get the indices of the positive and negative reviews
        positive_indices = [i for i in range(len(metacritic_probabilities)) if metacritic_probabilities[i] > 0.5]
        negative_indices = [i for i in range(len(metacritic_probabilities)) if metacritic_probabilities[i] <= 0.5]
        # Choose a random positive and negative review
        positive_review = metacritic_reviews_df.iloc[random.choice(positive_indices)]['review']
        negative_review = metacritic_reviews_df.iloc[random.choice(negative_indices)]['review']
        # Calculate the sentiment score
        metacritic_sentiment_score = len(positive_indices) / (len(positive_indices) + len(negative_indices))
    else:
        # Return 0 as the score if the dataframe is empty
        metacritic_sentiment_score = 0
        positive_review = ""
        negative_review = ""

    return {
        "imdb": imdb_sentiment_score,
        "metacritic": metacritic_sentiment_score,
        "positive_review": positive_review,
        "negative_review": negative_review
    }


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        movie_title = request.form["movie_title"]
        print(f"Movie title submitted: {movie_title}")
        sentiment_scores = analyze_movie_sentiment(movie_title)
        print(sentiment_scores)  # Add this line to print the sentiment_scores dictionary
        return render_template("results.html", movie_title=movie_title, sentiment_scores=sentiment_scores)
    print("Rendering index.html")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
    print("Flask server running...")
