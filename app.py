from flask import Flask, render_template, request, redirect, url_for
import os
import re
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Folder where the documents are stored
DOCUMENTS_FOLDER = "static/dataset_final"  # Ensure this folder exists

# Load necessary NLTK resources
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# Load documents from folder
def load_documents():
    documents = {}
    doc_id_to_filename = {}
    try:
        for i, filename in enumerate(os.listdir(DOCUMENTS_FOLDER)):
            if filename.endswith(".txt"):
                with open(
                    os.path.join(DOCUMENTS_FOLDER, filename), "r", encoding="utf-8"
                ) as f:
                    content = f.read()
                    documents[i] = content
                    doc_id_to_filename[i] = filename
    except FileNotFoundError:
        print(f"Error: Directory '{DOCUMENTS_FOLDER}' not found.")
        raise
    return documents, doc_id_to_filename


# Preprocess text: lowercase, remove special characters, remove stopwords, lemmatize
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(
        r"[^a-zA-Z0-9\s]", "", text
    )  # Remove special characters and punctuation
    tokens = word_tokenize(text)  # Tokenize the text
    cleaned_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
    ]
    return cleaned_tokens


# Build vocabulary (unique terms across all documents)
def build_vocab(documents):
    vocab = set()
    for doc in documents.values():
        vocab.update(clean_text(doc))
    return sorted(vocab)


# Calculate term frequency (TF)
def term_frequency(term, document):
    return document.count(term) / len(document)


# Calculate inverse document frequency (IDF)
def inverse_document_frequency(term, all_documents):
    num_docs_containing_term = sum(1 for doc in all_documents if term in doc)
    return math.log(len(all_documents) / (1 + num_docs_containing_term))


# Compute TF-IDF vector for a document
def compute_tfidf(document, all_documents, vocab):
    tfidf_vector = []
    for term in vocab:
        tf = term_frequency(term, document)
        idf = inverse_document_frequency(term, all_documents)
        tfidf_vector.append(tf * idf)
    return np.array(tfidf_vector)


# Calculate TF-IDF vectors for all documents
def calculate_tfidf_vectors(documents, vocab):
    tokenized_docs = {i: clean_text(doc) for i, doc in documents.items()}
    doc_tfidf_vectors = {
        i: compute_tfidf(doc, tokenized_docs.values(), vocab)
        for i, doc in tokenized_docs.items()
    }
    return doc_tfidf_vectors


# Load documents and prepare the TF-IDF vectors
documents, doc_id_to_filename = load_documents()
vocab = build_vocab(documents)
doc_tfidf_vectors = calculate_tfidf_vectors(documents, vocab)


# Evaluation Metric Functions
def precision_at_k(relevant_docs, retrieved_docs, k):
    relevant_retrieved = set(relevant_docs) & set(retrieved_docs[:k])
    return len(relevant_retrieved) / k if k > 0 else 0


def recall_at_k(relevant_docs, retrieved_docs, k):
    relevant_retrieved = set(relevant_docs) & set(retrieved_docs[:k])
    return len(relevant_retrieved) / len(relevant_docs) if len(relevant_docs) > 0 else 0


def average_precision(relevant_docs, retrieved_docs):
    score = 0.0
    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            score += precision_at_k(relevant_docs, retrieved_docs, i + 1)
    return score / len(relevant_docs) if relevant_docs else 0


def ndcg_at_k(relevant_docs, retrieved_docs, k):
    dcg = 0.0
    idcg = sum(
        1 / math.log2(i + 2) for i in range(min(k, len(relevant_docs)))
    )  # Ideal DCG
    for i in range(min(k, len(retrieved_docs))):
        if retrieved_docs[i] in relevant_docs:
            dcg += 1 / math.log2(i + 2)  # Calculate DCG
    return dcg / idcg if idcg > 0 else 0


# Search functionality
@app.route("/", methods=["GET", "POST"])
def search():
    results = None
    query = None
    message = None

    if request.method == "POST":
        query = request.form["query"]  # Get the user's query
        tokenized_query = clean_text(query)  # Preprocess the query
        query_vector = compute_tfidf(
            tokenized_query, documents.values(), vocab
        )  # Compute TF-IDF for the query

        # Calculate cosine similarity between query and all document vectors
        similarities = {
            i: cosine_similarity([query_vector], [doc_vector])[0][0]
            for i, doc_vector in doc_tfidf_vectors.items()
        }

        # Sort documents by similarity score
        ranked_docs = sorted(
            similarities.items(), key=lambda item: item[1], reverse=True
        )

        # Filter documents with similarity > 0
        top_docs_with_scores = [
            (doc_id_to_filename[doc_id].replace(".txt", ""), score)
            for doc_id, score in ranked_docs
            if score > 0
        ]

        if not top_docs_with_scores:
            message = "No results matched your search."
        else:
            results = top_docs_with_scores  # Display all relevant results

            # Automatically determine relevant documents from the output
            relevant_docs = [
                filename
                for filename, _ in results
                if any(term in filename.lower() for term in tokenized_query)
            ]

            retrieved_docs = [filename for filename, _ in top_docs_with_scores]

            # Evaluate metrics
            precision = precision_at_k(
                relevant_docs, retrieved_docs, k=len(retrieved_docs)
            )  # Use total number of retrieved docs
            recall = recall_at_k(relevant_docs, retrieved_docs, k=len(retrieved_docs))
            map_score = average_precision(relevant_docs, retrieved_docs)
            ndcg = ndcg_at_k(relevant_docs, retrieved_docs, k=len(retrieved_docs))

            # Print metrics in the terminal
            print(f"Query: '{query}'")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"MAP: {map_score:.4f}")
            print(f"nDCG: {ndcg:.4f}")

    return render_template("index.html", query=query, docs=results, message=message)


# Article route to display the document content
@app.route("/article/<filename>/<query>")
def article(filename, query):
    # Replace underscores with spaces when accessing the file
    filename = filename.replace("_", " ")
    filepath = os.path.join(DOCUMENTS_FOLDER, f"{filename}.txt")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return render_template(
            "article.html", filename=filename, content=content, query=query
        )
    except FileNotFoundError:
        return "Document not found", 404


if __name__ == "__main__":
    app.run(debug=True)
