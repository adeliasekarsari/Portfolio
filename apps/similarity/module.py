import torch
from fuzzywuzzy import fuzz
from sklearn.metrics import jaccard_score
from Levenshtein import distance as levenshtein_distance
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle

nltk.download('stopwords')
nltk.download('wordnet')
model = pickle.load(open('apps/similarity/model_v1.pkl', 'rb'))


def fuzzy_similarity(name1, name2):
    return fuzz.ratio(name1.lower(), name2.lower()) / 100.0

def jaccard_similarity(name1, name2):
    set1 = set(name1.lower().split())
    set2 = set(name2.lower().split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

def tfidf_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0]

def levenshtein_similarity(text1, text2):
    max_len = max(len(text1), len(text2))
    return 1 - levenshtein_distance(text1, text2) / max_len


def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z\s]', '', text)


def remove_extra_whitespaces(text):
    return re.sub(r'\s+', ' ', text).strip()

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words])


def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def preprocess_text(text):
    text = to_lowercase(text)
    text = remove_special_characters(text)
    text = remove_extra_whitespaces(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

def to_lowercase(text):
    return text.lower()

def processing_similarity(text1, text2):
    process_name1 = preprocess_text(text1)
    process_name2 = preprocess_text(text2)

    # Replace these with the actual similarity values calculated from the 4 methods
    similarity_features = {
        'fuzzy': fuzzy_similarity(process_name1, process_name2),
        'jaccard': jaccard_similarity(process_name1, process_name2),
        'tfidf_similarity': tfidf_similarity(process_name1, process_name2),
        'levenshtein_similarity': levenshtein_similarity(process_name1, process_name2)
    }

    # Convert similarity features into a DataFrame (or a format compatible with your model)
    input_data = pd.DataFrame([similarity_features])

    # Step 3: Make a prediction
    prediction = model.predict(input_data)
    return prediction, similarity_features



