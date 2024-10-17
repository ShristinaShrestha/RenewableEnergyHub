import math
from sklearn.metrics.pairwise import cosine_similarity

# Define evaluation functions


def precision_at_k(relevant_docs, retrieved_docs, k):
    return len(set(relevant_docs) & set(retrieved_docs[:k])) / k


def recall_at_k(relevant_docs, retrieved_docs, k):
    return len(set(relevant_docs) & set(retrieved_docs[:k])) / len(relevant_docs)


def average_precision(relevant_docs, retrieved_docs):
    precisions = []
    num_relevant = 0
    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            num_relevant += 1
            precisions.append(num_relevant / (i + 1))
    return sum(precisions) / len(relevant_docs)


def dcg_at_k(relevant_docs, retrieved_docs, k):
    dcg = 0.0
    for i in range(k):
        if retrieved_docs[i] in relevant_docs:
            dcg += 1 / math.log2(i + 2)
    return dcg


def idcg_at_k(k):
    idcg = 0.0
    for i in range(k):
        idcg += 1 / math.log2(i + 2)
    return idcg


def ndcg_at_k(relevant_docs, retrieved_docs, k):
    dcg = dcg_at_k(relevant_docs, retrieved_docs, k)
    idcg = idcg_at_k(k)
    return dcg / idcg if idcg > 0 else 0
