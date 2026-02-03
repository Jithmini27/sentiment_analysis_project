import os
import re
import pickle
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "static", "model", "model.pickle")
VOCAB_PATH = os.path.join(BASE_DIR, "static", "model", "vocabulary.txt")

# ----------------------------
# Load model
# ----------------------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ----------------------------
# Load vocabulary (strip spaces!)
# ----------------------------
vocab = pd.read_csv(VOCAB_PATH, header=None)
tokens = vocab[0].astype(str).str.strip().tolist()
tokens = [t for t in tokens if t]

# Stopwords
stop_words = set(stopwords.words("english"))

# ----------------------------
# Preprocessing (MATCH YOUR NOTEBOOK)
# ----------------------------
def preprocessing(text: str) -> str:
    t = str(text)

    # lowercase
    t = " ".join(word.lower() for word in t.split())

    # remove URLs
    t = re.sub(r"http\S+|www\S+", "", t)

    # remove punctuation (keep words/spaces)
    t = re.sub(r"[^\w\s]", "", t)

    # remove numbers
    t = re.sub(r"\d+", "", t)

    # remove stopwords
    t = " ".join(word for word in t.split() if word not in stop_words)

    return t

# ----------------------------
# Vectorizer (same as notebook)
# ----------------------------
def vectorizer(ds, vocabulary=tokens):
    vectorized_list = []

    for sentence in ds:
        sentence_lst = np.zeros(len(vocabulary), dtype=np.float32)
        words = sentence.split()

        for i in range(len(vocabulary)):
            if vocabulary[i] in words:
                sentence_lst[i] = 1.0

        vectorized_list.append(sentence_lst)

    return np.asarray(vectorized_list, dtype=np.float32)

# ----------------------------
# Prediction
# 0 = Positive, 1 = Negative (your mapping)
# ----------------------------
def get_prediction(raw_text: str) -> str:
    clean = preprocessing(raw_text)
    X = vectorizer([clean], tokens)
    pred = int(model.predict(X)[0])

    return "negative" if pred == 1 else "positive"
