import streamlit as st
import pandas as pd
from keras.models import load_model
import requests
from imdb import IMDb
from bs4 import BeautifulSoup
from preprocessing import preprocess
from wordcloud import WordCloud
import matplotlib.pyplot as plt


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

    def get_metacritic_reviews(movie_title, max_pages=4):
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
    
    def generate_wordcloud(reviews):
        text = " ".join(review for review in reviews)
        wordcloud = WordCloud(width=800, height=800, background_color="black", stopwords=None, contour_width=3, contour_color="steelblue").generate(text)

        return wordcloud
    
    data = {"review": imdb_reviews}
    imdb_reviews_df = pd.DataFrame(data)

    # Fetch Metacritic reviews
    metacritic_reviews_df = get_metacritic_reviews(movie_title)
    print(f"Fetched {len(metacritic_reviews_df)} Metacritic reviews")

     # Preprocess the IMDb reviews
    imdb_reviews_preprocessed = preprocess(imdb_reviews_df['review'])
    print(f"Preprocessed {len(imdb_reviews_preprocessed)} IMDb reviews")

    metacritic_reviews_preprocessed = preprocess(metacritic_reviews_df['review'])
    print(f"Preprocessed {len(metacritic_reviews_preprocessed)} Metacritic reviews")

    # Use the model to predict sentiment scores
    # twitter_scores = model.predict(twitter_reviews_preprocessed)
    imdb_probabilities = model.predict(imdb_reviews_preprocessed)
    metacritic_probabilities = model.predict(metacritic_reviews_preprocessed)

    # Filter negative Metacritic reviews (with a sentiment score of 0.5 or less)
    negative_metacritic_reviews = [review for review, prob in zip(metacritic_reviews_df['review'], metacritic_probabilities) if prob <= 0.5]
    positive_metacritic_reviews = [review for review, prob in zip(metacritic_reviews_df['review'], metacritic_probabilities) if prob > 0.5]

    # Generate word cloud for negative Metacritic reviews
    metacritic_negative_wordcloud = generate_wordcloud(negative_metacritic_reviews)

    # Generate word cloud for negative Metacritic reviews
    metacritic_positive_wordcloud = generate_wordcloud(positive_metacritic_reviews)

    # Sort probabilities in descending order
    metacritic_sorted_indices = sorted(range(len(metacritic_probabilities)), key=lambda k: metacritic_probabilities[k], reverse=True)

    # Get the index of the positive review with the highest probability
    positive_index = next((i for i in metacritic_sorted_indices if metacritic_probabilities[i] > 0.5), None)

    # Get the index of the negative review with the lowest probability
    negative_index = next((i for i in reversed(metacritic_sorted_indices) if metacritic_probabilities[i] <= 0.5), None)

    # Check if both indices were found
    if positive_index is not None and negative_index is not None:
        positive_review = metacritic_reviews_df.iloc[positive_index]['review']
        negative_review = metacritic_reviews_df.iloc[negative_index]['review']
        # Calculate the sentiment score
        metacritic_sentiment_score = (len(metacritic_sorted_indices) - negative_index) / len(metacritic_sorted_indices)
    else:
        # Return 0 as the score if either index is None
        metacritic_sentiment_score = 0.0
        print("Positive or negative index not found")
    
    # Count the number of positive predictions
    imdb_positive_count = (imdb_probabilities).sum()
    # Calculate the sentiment score
    imdb_sentiment_score = imdb_positive_count / len(imdb_probabilities)

    # Modify the return statement to include the negative Metacritic word cloud
    return {
        "imdb": imdb_sentiment_score,
        "metacritic": metacritic_sentiment_score,
        "positive_review": positive_review,
        "negative_review": negative_review,
        "metacritic_negative_wordcloud": metacritic_negative_wordcloud,
        "metacritic_positive_wordcloud": metacritic_positive_wordcloud
    }

def index():
    st.title("Movie Review Sentiment Analysis")
    movie_title = st.text_input("Enter a movie title")

    if st.button("Analyze Sentiment"):
        if not movie_title:
            st.error("Please enter a movie title")
        else:
            with st.spinner("Analyzing..."):
                sentiment_scores = analyze_movie_sentiment(movie_title)
                st.write(f"Sentiment analysis for {movie_title.title()}:")

                
                imdb_score = f"{sentiment_scores['imdb']*100:.2f}%"
                imdb_score_str = f"IMDb Score: {imdb_score}"
                imdb_score_dial = st.progress(sentiment_scores['imdb'], text = 'IMDb Score: '+ imdb_score)
                
                metacritic_score = f"{sentiment_scores['metacritic']*100:.2f}%"
                metacritic_score_str = f"Metacritic Score: {metacritic_score}"
                metacritic_score_dial = st.progress(sentiment_scores['metacritic'], text = 'Metacritic Score: '+ metacritic_score)
                
                st.subheader("Positive review:")
                st.success(sentiment_scores['positive_review'])
                
                st.subheader("Negative review:")
                st.error(sentiment_scores['negative_review'])

                st.subheader("Word Cloud for Positive Metacritic Reviews:")
                st.image(sentiment_scores["metacritic_positive_wordcloud"].to_array(), use_column_width=True)

                st.subheader("Word Cloud for Negative Metacritic Reviews:")
                st.image(sentiment_scores["metacritic_negative_wordcloud"].to_array(), use_column_width=True)



if __name__ == '__main__':
    index()