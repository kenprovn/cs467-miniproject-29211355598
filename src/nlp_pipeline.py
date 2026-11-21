"""
NLP Preprocessing and TF-IDF Matching Pipeline (MP-C06)
Student: Nguyễn Minh Khanh (29211355598)
Course: CS467 Advanced Artificial Intelligence - CS467C

Implements:
- Text normalization (lowercasing, punctuation removal, whitespace cleaning)
- Stop-word removal and tokenization
- Scikit-learn TF-IDF Vectorization with unigrams and bigrams
- Cosine similarity computation
- Salient keyword extraction between queries and eligibility profiles
- Free-text query vectorization using the same fitted model
- Item-to-item similarity matching
"""

import os
import re
import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_text(text: str) -> str:
    """
    Standardizes input text:
    1. Converts to string and lowercase.
    2. Removes punctuation and special characters.
    3. Normalizes whitespace.
    4. Filters out common stop words while preserving core domain terms.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # 1. Lowercase
    text = text.lower()

    # 2. Replace punctuation with space
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # 3. Collapse multiple whitespaces
    tokens = text.split()

    # 4. Filter stop words
    filtered_tokens = [tok for tok in tokens if tok not in ENGLISH_STOP_WORDS and len(tok) > 1]

    return " ".join(filtered_tokens)


class ScholarshipNLPPipeline:
    """
    NLP Pipeline for scholarship indexing, retrieval, and similarity analysis.
    Uses TF-IDF representation and Cosine Similarity.
    """

    def __init__(self, items_df: pd.DataFrame = None):
        self.items_df = items_df
        self.vectorizer = TfidfVectorizer(
            preprocessor=preprocess_text,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            norm="l2",
        )
        self.tfidf_matrix = None
        self.feature_names = None
        self.item_id_to_idx = {}
        self.idx_to_item_id = {}

        if items_df is not None:
            self.fit(items_df)

    def _create_rich_corpus(self, df: pd.DataFrame) -> list:
        """
        Combines grant_name, degree_level, country, focus_tags, and eligibility_text
        into a dense semantic document per scholarship.
        """
        corpus = []
        for _, row in df.iterrows():
            text_parts = [
                row.get("grant_name", ""),
                row.get("degree_level", ""),
                row.get("country", ""),
                row.get("focus_tags", ""),
                row.get("eligibility_text", ""),
            ]
            doc = " ".join([str(p) for p in text_parts if pd.notna(p)])
            corpus.append(doc)
        return corpus

    def fit(self, items_df: pd.DataFrame):
        """Fits the TF-IDF vectorizer across the scholarship catalog."""
        self.items_df = items_df.copy().reset_index(drop=True)
        self.item_id_to_idx = {row["item_id"]: idx for idx, row in self.items_df.iterrows()}
        self.idx_to_item_id = {idx: row["item_id"] for idx, row in self.items_df.iterrows()}

        corpus = self._create_rich_corpus(self.items_df)
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())
        return self

    def extract_matching_keywords(self, query_vec, doc_idx: int, top_n: int = 5) -> list:
        """
        Identifies the top shared TF-IDF terms between a query and a scholarship document.
        Calculates coordinate-wise product w_i = q_i * d_i to extract salient overlapping terms.
        """
        doc_vec = self.tfidf_matrix[doc_idx]
        # Multiply sparse query vector with sparse document vector
        overlap = query_vec.multiply(doc_vec)
        non_zero_indices = overlap.nonzero()[1]

        if len(non_zero_indices) > 0:
            scores = overlap.data
            sorted_order = np.argsort(scores)[::-1]
            top_terms = [self.feature_names[non_zero_indices[i]] for i in sorted_order[:top_n]]
            return top_terms

        # Fallback: return top terms of document if exact overlap is sparse
        doc_data = doc_vec.data
        doc_indices = doc_vec.nonzero()[1]
        if len(doc_indices) > 0:
            sorted_doc = np.argsort(doc_data)[::-1]
            return [self.feature_names[doc_indices[i]] for i in sorted_doc[:top_n]]

        return []

    def search_by_query(self, query_text: str, top_k: int = 5) -> list:
        """
        Vectorizes free-text query with the fitted TF-IDF model and retrieves Top-K
        scholarships ranked by Cosine Similarity, along with matching keywords.
        """
        if self.tfidf_matrix is None:
            raise ValueError("Pipeline has not been fitted with scholarship data yet.")

        # Vectorize using the EXACT same fitted vectorizer
        query_vec = self.vectorizer.transform([query_text])

        # Compute cosine similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            item = self.items_df.iloc[idx]
            keywords = self.extract_matching_keywords(query_vec, idx, top_n=5)
            results.append({
                "item_id": item["item_id"],
                "grant_name": item["grant_name"],
                "country": item["country"],
                "degree_level": item["degree_level"],
                "similarity_score": round(score, 4),
                "matching_keywords": keywords,
                "focus_tags": item["focus_tags"],
            })

        return results

    def find_similar_items(self, item_id: str, top_k: int = 5) -> list:
        """
        Calculates item-to-item similarity across the TF-IDF feature space.
        Strictly excludes the query item itself from the results.
        """
        if item_id not in self.item_id_to_idx:
            raise KeyError(f"Item ID '{item_id}' not found in catalog.")

        query_idx = self.item_id_to_idx[item_id]
        query_vec = self.tfidf_matrix[query_idx]

        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Sort descending
        sorted_indices = np.argsort(similarities)[::-1]

        # Filter out self
        similar_indices = [idx for idx in sorted_indices if idx != query_idx][:top_k]

        results = []
        for idx in similar_indices:
            score = float(similarities[idx])
            item = self.items_df.iloc[idx]
            keywords = self.extract_matching_keywords(query_vec, idx, top_n=5)
            results.append({
                "item_id": item["item_id"],
                "grant_name": item["grant_name"],
                "country": item["country"],
                "degree_level": item["degree_level"],
                "similarity_score": round(score, 4),
                "matching_keywords": keywords,
                "focus_tags": item["focus_tags"],
            })

        return results

    def score_unseen_item(self, new_item: dict, query_text: str) -> dict:
        """
        Evaluates a newly announced scholarship (cold-start item with zero ratings)
        against a free-text profile or query using the fitted TF-IDF model.
        """
        doc = " ".join([
            str(new_item.get("grant_name", "")),
            str(new_item.get("degree_level", "")),
            str(new_item.get("country", "")),
            str(new_item.get("focus_tags", "")),
            str(new_item.get("eligibility_text", "")),
        ])

        new_item_vec = self.vectorizer.transform([doc])
        query_vec = self.vectorizer.transform([query_text])

        sim = float(cosine_similarity(query_vec, new_item_vec)[0, 0])

        # Overlap keywords
        overlap = query_vec.multiply(new_item_vec)
        non_zero = overlap.nonzero()[1]
        matching_keywords = []
        if len(non_zero) > 0:
            scores = overlap.data
            sorted_order = np.argsort(scores)[::-1]
            matching_keywords = [self.feature_names[non_zero[i]] for i in sorted_order[:5]]

        return {
            "grant_name": new_item.get("grant_name", ""),
            "country": new_item.get("country", ""),
            "degree_level": new_item.get("degree_level", ""),
            "similarity_score": round(sim, 4),
            "matching_keywords": matching_keywords,
        }


if __name__ == "__main__":
    # Self-test for NLP pipeline
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    items_path = os.path.join(base_dir, "data", "items.csv")
    df = pd.read_csv(items_path)

    pipeline = ScholarshipNLPPipeline(df)
    test_query = "undergraduate computer science student seeking AI research summer exchange"
    print(f"Testing Query: '{test_query}'")
    results = pipeline.search_by_query(test_query, top_k=5)
    for r in results:
        print(f" - [{r['item_id']}] {r['grant_name']} ({r['country']}, {r['degree_level']}): score={r['similarity_score']} | matching={r['matching_keywords']}")

    print("\nTesting Item-to-Item Similarity for SCH_01:")
    similar = pipeline.find_similar_items("SCH_01", top_k=5)
    for s in similar:
        print(f" - [{s['item_id']}] {s['grant_name']}: score={s['similarity_score']}")
