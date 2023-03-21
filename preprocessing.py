import re
import nltk
import pandas as pd
from nltk.corpus import stopwords, words
from nltk.stem.wordnet import WordNetLemmatizer as wnl
from nltk.stem import SnowballStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_words = 20000
maxlen = 500
stop_words = set(stopwords.words('english'))
# Get a list of valid English words
word_list = words.words()

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'
    elif treebank_tag.startswith('V'):
        return 'v'
    elif treebank_tag.startswith('N'):
        return 'n'
    elif treebank_tag.startswith('R'):
        return 'r'
    else:
        return None

def replace_non_ascii(str_a):
    return str_a.translate(str.maketrans('', '', ''.join([chr(i) for i in range(128, 256)])))

def clean_str(string):
    return string.replace("\\", "").replace("'", "").replace('"', '').strip().lower()

# Define a function to lemmatize a sentence using WordNetLemmatizer
def lemmatize_sentence(sentence, word_list, stop_words):
    # Create a WordNetLemmatizer object
    lemmatizer = SnowballStemmer(language='english')
    
    # Tokenize the sentence and get the part of speech tag for each token
    tagged = nltk.pos_tag(nltk.word_tokenize(sentence.lower()))
    
    # Convert the part of speech tags to WordNet format
    wordnet_tagged = [(word, get_wordnet_pos(pos)) for (word, pos) in tagged]
    
    # Lemmatize each valid token in the sentence and remove stop words and non-valid words
    lemmatized_sentence = [
        lemmatizer.stem(token)
        for token, pos in wordnet_tagged
        if token.lower() in word_list and token.lower() not in stop_words
    ]
    
    # Join the lemmatized tokens back into a sentence
    return ' '.join(lemmatized_sentence)

def apply_lemmatization(df):
    # Apply the lemmatize_sentence function to the 'review' column of the DataFrame
    return df['review'].apply(lambda x: lemmatize_sentence(x, word_list, stop_words))

def preprocess(df):
    df['review'] = df['review'].apply(replace_non_ascii)
    df['review'] = df['review'].apply(clean_str)
    lemmatized_reviews = apply_lemmatization(df)
    lemmatized_reviews_df = pd.DataFrame({'review': lemmatized_reviews})
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(lemmatized_reviews_df['review'])
    sequences = tokenizer.texts_to_sequences(lemmatized_reviews_df['review'])
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)
    return padded_sequences
