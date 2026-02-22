# =========================================================
# Assignment 4 – Information Retrieval
# =========================================================


# Tasks:
# 1. Posting Lists Creation (manual inverted index)
# 2. Cosine Similarity Computation (Manual COSINESCORe Algorithm, sparse)
# 3. Efficiency Strategy for Faster Retrieval


import pandas as pd
import re
import zipfile
import os
import time
from collections import defaultdict
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk


# =========================================================
# TASK 1 – Load, Unzip, and Create Posting Lists
# =========================================================

# Download stopwords if needed
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Custom Albanian stopwords
albanian_stopwords = {
    "dhe", "të", "në", "është", "për", "me", "nga", "si", "një", "që", "ai", "ajo", "ne", "ju", "ata",
    "atë", "kjo", "ky", "tek", "jam", "ke", "janë", "ishte", "kam", "ka", "kanë", "kemi",
    "por", "ose", "edhe", "pra", "po", "jo", "kur", "pasi", "sepse", "mbi", "pa", "më", "tani",
    "shumë", "vetëm", "ashtu", "atëherë", "duhet", "mund", "nuk", "sipas", "këtë", "këtij"
}

# ---------- Unzip Dataset ----------
zip_path = "002-news-2025.csv.zip"
extract_path = "datasets/"
csv_name = "002-news-2025.csv"

if not os.path.exists(os.path.join(extract_path, csv_name)):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
        print(" Dataset unzipped successfully.")
else:
    print(" Dataset already extracted.")

# ---------- Load Dataset ----------
csv_path = os.path.join(extract_path, csv_name)
df = pd.read_csv(csv_path)
df = df.dropna(subset=["content"])
documents = df["content"].tolist()
print(f"Loaded {len(documents)} documents.\n")

# ---------- Preprocessing ----------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zçë\s]", " ", text)
    tokens = [w for w in text.split() if w not in albanian_stopwords and len(w) > 1]
    return tokens

# ---------- Create Posting Lists ----------
posting_lists = defaultdict(set)

for doc_id, text in enumerate(documents):
    tokens = preprocess(text)
    for token in set(tokens):
        posting_lists[token].add(doc_id)

print(f" Number of unique terms: {len(posting_lists)}")
print(" Sample posting list for 'lajme':", list(posting_lists.get("lajme", []))[:10])
print(" Task 1 completed.\n")

# =========================================================
# TASK 2 – Cosine Similarity Computation (Manual COSINESCORe Sparse)
# =========================================================

vectorizer = TfidfVectorizer(
    tokenizer=preprocess,
    lowercase=True,
    stop_words=None,
    token_pattern=None
)

tfidf_matrix = vectorizer.fit_transform(documents)
terms = vectorizer.get_feature_names_out()

def cosine_score_sparse_show_docs(query, k=5):
    """
    Implements COSINESCORe(q) algorithm (IR Book p.125)
    Shows both document IDs and content.
    """
    start_time = time.time()
    N = len(documents)
    Scores = np.zeros(N)

    # Document lengths (sparse)
    Length = np.sqrt(tfidf_matrix.multiply(tfidf_matrix).sum(axis=1)).A1

    # Query preprocessing
    query_tokens = preprocess(query)
    query_vec = vectorizer.transform([" ".join(query_tokens)])
    vocab_index = vectorizer.vocabulary_

    # Algorithm mapping to IR book
    for t in query_tokens:                  # Step 3–4
        if t not in vocab_index:
            continue
        t_index = vocab_index[t]
        w_tq = query_vec[0, t_index]

        postings = posting_lists.get(t, [])  # Step 5
        for d in postings:                   # Step 6
            w_td = tfidf_matrix[d, t_index]
            Scores[d] += w_td * w_tq

    # Normalize Step 7–9
    nonzero = Length != 0
    Scores[nonzero] = Scores[nonzero] / Length[nonzero]

    # Return top K Step 10
    top_indices = np.argsort(Scores)[::-1][:k]
    results = [(i, Scores[i], documents[i]) for i in top_indices if Scores[i] > 0]

    end_time = time.time()
    print(f" Task 2 computation time: {end_time - start_time:.4f} sec")
    return results

# ---------- Example Query ----------
query = "ekonomia dhe tregu në Shqipëri"
print("🔹 Task 2 Results (Manual COSINESCORe Sparse):")
results = cosine_score_sparse_show_docs(query, k=5)
for doc_id, score, content in results:
    print(f"\nDoc ID {doc_id} | Score: {score:.4f}\nContent: {content[:300]}...")  # show first 300 chars

print("\n Task 2 completed.\n")

# =========================================================
# TASK 3 – Efficiency Strategy (IR Book p.126–128)
# =========================================================
# Strategy: Only consider docs containing ≥ half of query terms

def cosine_score_filtered_show_docs(query, k=5):
    query_tokens = preprocess(query)
    min_terms = max(1, len(query_tokens) // 2)

    # Count query terms per doc
    doc_count = defaultdict(int)
    for t in query_tokens:
        for d in posting_lists.get(t, []):
            doc_count[d] += 1

    # Filter docs
    candidate_docs = [d for d, c in doc_count.items() if c >= min_terms]
    print(f" Docs after filtering: {len(candidate_docs)} (≥ {min_terms} query terms)")

    Scores = np.zeros(len(documents))
    Length = np.sqrt(tfidf_matrix.multiply(tfidf_matrix).sum(axis=1)).A1
    vocab_index = vectorizer.vocabulary_
    query_vec = vectorizer.transform([" ".join(query_tokens)])

    start = time.time()
    for t in query_tokens:
        if t not in vocab_index:
            continue
        t_index = vocab_index[t]
        w_tq = query_vec[0, t_index]
        for d in candidate_docs:
            w_td = tfidf_matrix[d, t_index]
            Scores[d] += w_td * w_tq

    for d in candidate_docs:
        if Length[d] != 0:
            Scores[d] = Scores[d] / Length[d]

    top_indices = np.argsort(Scores)[::-1][:k]
    top_docs = [(i, Scores[i], documents[i]) for i in top_indices if Scores[i] > 0]
    end = time.time()
    print(f" Task 3 filtered computation time: {end - start:.4f} sec")
    return top_docs

# ---------- Task 3 Execution ----------
print("Task 3 Results (Filtered Strategy):")
filtered_results = cosine_score_filtered_show_docs(query, k=5)
for doc_id, score, content in filtered_results:
    print(f"\nDoc ID {doc_id} | Score: {score:.4f}\nContent: {content[:300]}...")  # first 300 chars

print("\n Task 3 completed.")
