
from transformers import BertTokenizer, BertModel
import torch
from fuzzywuzzy import fuzz
from sklearn.metrics import jaccard_score
import numpy as np

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(name):
    inputs = tokenizer(name, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding

def cosine_similarity(embedding1, embedding2):
    return torch.nn.functional.cosine_similarity(embedding1, embedding2).item()

def fuzzy_similarity(name1, name2):
    return fuzz.ratio(name1.lower(), name2.lower()) / 100.0

def jaccard_similarity(name1, name2):
    set1 = set(name1.lower().split())
    set2 = set(name2.lower().split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

def combined_similarity(name1, name2, weights=None):
    if weights is None:
        weights = [0.4, 0.3, 0.3]  # Adjust these weights as needed
    
    # Get embeddings
    embedding1 = get_bert_embedding(name1)
    embedding2 = get_bert_embedding(name2)
    
    # Calculate individual similarities
    cos_sim = cosine_similarity(embedding1, embedding2)
    fuzzy_sim = fuzzy_similarity(name1, name2)
    jac_sim = jaccard_similarity(name1, name2)
    
    # Combine similarities using the weights
    combined_score = (
        weights[0] * cos_sim +
        weights[1] * fuzzy_sim +
        weights[2] * jac_sim
    )
    
    return combined_score


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

