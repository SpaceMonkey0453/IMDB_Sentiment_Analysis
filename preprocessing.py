import re
import nltk
import pickle
from nltk.corpus import stopwords, words
from nltk.stem import SnowballStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

max_words = 20000
maxlen = 500
stop_words = set(stopwords.words('english'))
# Get a list of valid English words
word_list = words.words()
# Define a function to replace all non-ASCII characters with a space
def replace_non_ascii_regex(str_a):
     # Remove non-ASCII characters
    str_a = re.sub('[^\x00-\x7F]+', ' ', str_a)
    
    # Remove non-alphanumeric and non-whitespace characters
    str_a = re.sub('[^0-9a-zA-Z\s]+', ' ', str_a)
    return str_a

# Define a function to clean a string by removing unnecessary characters
def clean_str(string):
    return string.replace("\\", "").replace("'", "").replace('"', '').strip().lower()

# Define a function to lemmatize a sentence using SnowballStemmer
def lemmatize_sentence(sentence, word_list, stop_words):
    # Create a SnowballStemmer object
    lemmatizer = SnowballStemmer(language='english')
    
    # Tokenize the sentence into words
    words = nltk.word_tokenize(sentence.lower())
    
    # Lemmatize each valid word in the sentence and remove stop words and non-valid words
    lemmatized_sentence = [
        lemmatizer.stem(word)
        for word in words
        if word.lower() in word_list and word.lower() not in stop_words
    ]
    
    # Join the lemmatized words back into a sentence
    return ' '.join(lemmatized_sentence)

# Define a function to apply lemmatization to a Pandas DataFrame
def apply_lemmatization(df):
    # Apply the lemmatize_sentence function to the 'review' column of the DataFrame
    return df['review'].apply(lambda x: lemmatize_sentence(x, word_list, stop_words))

# Define a function to preprocess the reviews
def preprocess(reviews, tokenizer_path='tokenizer.pickle', max_len=500):
    # Load the saved tokenizer
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Preprocess the text
    preprocessed_reviews = []
    for review in reviews:
        review = replace_non_ascii_regex(review)
        review = clean_str(review)
        preprocessed_reviews.append(review)

    # Convert text to sequences of tokens using the loaded tokenizer
    sequences = tokenizer.texts_to_sequences(preprocessed_reviews)

    # Pad the sequences to a fixed length
    padded_sequences = pad_sequences(sequences, maxlen=max_len)

    return padded_sequences


