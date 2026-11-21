"""
Scholarship Recommendation Engine (MP-C06)
Student: Nguyễn Minh Khanh (29211355598)
Course: CS467 Advanced Artificial Intelligence - CS467C

Implements 3 recommendation methods:
1. Popularity Baseline: Bayesian weighted rating & average score filtering out user-rated items.
2. NLP Content-Based Filtering: TF-IDF vectorization & Cosine Similarity for student profiles,
   user rating history profiles, and item-to-item matching.
3. Hybrid Recommendation: Adaptive combination of NLP Content-Based and User-Based
   Collaborative Filtering with cold-start fallback handling.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.nlp_pipeline import ScholarshipNLPPipeline, preprocess_text


class PopularityRecommender:
    """
    Method 1: Popularity Baseline Recommender.
    Ranks items using Bayesian weighted rating:
        WR = (v / (v + m)) * R + (m / (v + m)) * C
    where:
        v = number of ratings for the item
        m = minimum rating count threshold (default: 2)
        R = average rating of the item
        C = mean rating across the entire dataset
    """

    def __init__(self, ratings_df: pd.DataFrame, items_df: pd.DataFrame, min_ratings: int = 2):
        self.ratings_df = ratings_df.copy()
        self.items_df = items_df.copy()
        self.min_ratings = min_ratings
        self.global_mean = self.ratings_df["relevance_rating"].mean()
        self._compute_popularity_table()

    def _compute_popularity_table(self):
        grouped = self.ratings_df.groupby("item_id")["relevance_rating"].agg(["count", "mean"]).reset_index()
        grouped.columns = ["item_id", "rating_count", "avg_rating"]

        m = self.min_ratings
        c = self.global_mean

        def bayesian_score(row):
            v = row["rating_count"]
            r = row["avg_rating"]
            if v == 0:
                return c
            return (v / (v + m)) * r + (m / (v + m)) * c

        grouped["popularity_score"] = grouped.apply(bayesian_score, axis=1)

        # Merge with items metadata
        self.item_stats = pd.merge(self.items_df, grouped, on="item_id", how="left")
        self.item_stats["rating_count"] = self.item_stats["rating_count"].fillna(0).astype(int)
        self.item_stats["avg_rating"] = self.item_stats["avg_rating"].fillna(c)
        self.item_stats["popularity_score"] = self.item_stats["popularity_score"].fillna(c)

    def recommend(self, user_id: str = None, top_k: int = 5, exclude_rated: bool = True) -> list:
        """Returns top_k popular scholarships, excluding items already rated by user_id."""
        rated_items = set()
        if user_id and exclude_rated and not self.ratings_df.empty:
            rated_items = set(self.ratings_df[self.ratings_df["user_id"] == user_id]["item_id"])

        candidates = self.item_stats[~self.item_stats["item_id"].isin(rated_items)].copy()
        candidates = candidates.sort_values(by=["popularity_score", "rating_count"], ascending=[False, False])

        top_candidates = candidates.head(top_k)
        results = []
        for _, row in top_candidates.iterrows():
            results.append({
                "item_id": row["item_id"],
                "grant_name": row["grant_name"],
                "country": row["country"],
                "degree_level": row["degree_level"],
                "score": round(float(row["popularity_score"]), 4),
                "avg_rating": round(float(row["avg_rating"]), 2),
                "rating_count": int(row["rating_count"]),
                "method": "Popularity Baseline",
            })
        return results


class ContentBasedRecommender:
    """
    Method 2: NLP Content-Based Filtering Recommender.
    Leverages TF-IDF and Cosine Similarity to compute recommendations from:
    - Free-text student query / profile
    - User's historical high ratings (user profile vector)
    - Item-to-item similarity
    """

    def __init__(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame = None):
        self.items_df = items_df.copy()
        self.ratings_df = ratings_df.copy() if ratings_df is not None else pd.DataFrame()
        self.pipeline = ScholarshipNLPPipeline(self.items_df)

    def recommend_for_query(self, query_text: str, top_k: int = 5) -> list:
        """Recommends scholarships based on a natural language student query."""
        results = self.pipeline.search_by_query(query_text, top_k=top_k)
        for r in results:
            r["method"] = "NLP Content-Based"
            r["score"] = r["similarity_score"]
        return results

    def recommend_similar_items(self, item_id: str, top_k: int = 5) -> list:
        """Finds most similar scholarships to a given scholarship, excluding itself."""
        results = self.pipeline.find_similar_items(item_id, top_k=top_k)
        for r in results:
            r["method"] = "NLP Content-Based (Item-to-Item)"
            r["score"] = r["similarity_score"]
        return results

    def build_user_profile_vector(self, user_id: str):
        """
        Constructs a pseudo-profile vector for a user by aggregating TF-IDF vectors
        of items the user rated favorably (rating >= 3.5, weighted by rating score).
        """
        if self.ratings_df.empty or user_id not in self.ratings_df["user_id"].values:
            return None

        user_ratings = self.ratings_df[self.ratings_df["user_id"] == user_id]
        favorable = user_ratings[user_ratings["relevance_rating"] >= 3.0]

        if favorable.empty:
            favorable = user_ratings  # fallback to all rated items

        profile_vec = np.zeros((1, self.pipeline.tfidf_matrix.shape[1]))
        total_weight = 0.0

        for _, row in favorable.iterrows():
            item_id = row["item_id"]
            if item_id in self.pipeline.item_id_to_idx:
                idx = self.pipeline.item_id_to_idx[item_id]
                weight = float(row["relevance_rating"])
                profile_vec += self.pipeline.tfidf_matrix[idx].toarray() * weight
                total_weight += weight

        if total_weight > 0:
            profile_vec /= total_weight
            norm = np.linalg.norm(profile_vec)
            if norm > 0:
                profile_vec /= norm

        return profile_vec

    def recommend_for_user(self, user_id: str, top_k: int = 5, exclude_rated: bool = True) -> list:
        """Recommends scholarships to an existing user based on their rated preferences."""
        rated_items = set()
        if user_id and exclude_rated and not self.ratings_df.empty:
            rated_items = set(self.ratings_df[self.ratings_df["user_id"] == user_id]["item_id"])

        user_vec = self.build_user_profile_vector(user_id)
        if user_vec is None:
            # Fallback if cold-start user
            return []

        # Cosine similarity between user profile vector and all scholarships
        similarities = cosine_similarity(user_vec, self.pipeline.tfidf_matrix).flatten()
        sorted_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in sorted_indices:
            item_id = self.pipeline.idx_to_item_id[idx]
            if exclude_rated and item_id in rated_items:
                continue

            item = self.items_df.iloc[idx]
            results.append({
                "item_id": item_id,
                "grant_name": item["grant_name"],
                "country": item["country"],
                "degree_level": item["degree_level"],
                "score": round(float(similarities[idx]), 4),
                "similarity_score": round(float(similarities[idx]), 4),
                "focus_tags": item["focus_tags"],
                "method": "NLP Content-Based",
            })
            if len(results) >= top_k:
                break

        return results


class CollaborativeFilteringRecommender:
    """
    User-Based Collaborative Filtering using Pearson / Cosine similarity on centered ratings.
    """

    def __init__(self, ratings_df: pd.DataFrame, items_df: pd.DataFrame):
        self.ratings_df = ratings_df.copy()
        self.items_df = items_df.copy()
        self._build_user_item_matrix()

    def _build_user_item_matrix(self):
        # Create user-item interaction matrix
        self.ui_matrix = self.ratings_df.pivot(index="user_id", columns="item_id", values="relevance_rating")
        self.all_users = list(self.ui_matrix.index)
        self.all_items = list(self.items_df["item_id"])

        # Compute user means (ignoring NaN)
        self.user_means = self.ui_matrix.mean(axis=1)

        # Center ratings
        centered = self.ui_matrix.sub(self.user_means, axis=0).fillna(0.0)

        # Compute user-user similarity matrix via cosine similarity on centered ratings (Pearson correlation equivalent)
        user_norms = np.linalg.norm(centered.values, axis=1, keepdims=True)
        user_norms[user_norms == 0] = 1e-9
        normalized_centered = centered.values / user_norms
        self.user_sim_matrix = pd.DataFrame(
            np.dot(normalized_centered, normalized_centered.T),
            index=self.all_users,
            columns=self.all_users,
        )

    def predict_rating(self, user_id: str, item_id: str) -> float:
        """Predicts relevance rating for (user_id, item_id). Returns normalized [0, 1] score."""
        if user_id not in self.ui_matrix.index:
            # Cold-start user
            return 0.5

        if item_id not in self.ui_matrix.columns:
            # Cold-start item
            return float(self.user_means.get(user_id, 3.0)) / 5.0

        user_mean = float(self.user_means[user_id])
        item_ratings = self.ui_matrix[item_id].dropna()

        if item_ratings.empty:
            return user_mean / 5.0

        sims = self.user_sim_matrix.loc[user_id, item_ratings.index]
        positive_mask = sims > 0

        if not positive_mask.any():
            return user_mean / 5.0

        pos_sims = sims[positive_mask]
        pos_ratings = item_ratings[positive_mask]
        pos_user_means = self.user_means[pos_ratings.index]

        numerator = np.sum(pos_sims * (pos_ratings - pos_user_means))
        denominator = np.sum(np.abs(pos_sims))

        if denominator == 0:
            pred = user_mean
        else:
            pred = user_mean + (numerator / denominator)

        # Clip predicted rating to [1.0, 5.0] and normalize to [0, 1]
        pred_clipped = max(1.0, min(5.0, pred))
        normalized_score = (pred_clipped - 1.0) / 4.0
        return normalized_score


class HybridRecommender:
    """
    Method 3: Hybrid Recommender.
    Harmonizes NLP Content-Based Filtering with User-Based Collaborative Filtering:
        Hybrid_Score = alpha * Score_CB + (1 - alpha) * Score_CF
    
    Design & Rationale:
    - alpha (default = 0.5): Balances semantic match against peer recommendation patterns.
    - Cold-start resilience: For cold-start items (0 ratings) or new users with no history,
      the recommender smoothly shifts alpha -> 1.0 (pure Content-Based), ensuring continuous operation.
    """

    def __init__(self, ratings_df: pd.DataFrame, items_df: pd.DataFrame, alpha: float = 0.5):
        self.ratings_df = ratings_df.copy()
        self.items_df = items_df.copy()
        self.alpha = alpha

        self.cb_recommender = ContentBasedRecommender(self.items_df, self.ratings_df)
        self.cf_recommender = CollaborativeFilteringRecommender(self.ratings_df, self.items_df)

    def recommend(self, user_id: str, top_k: int = 5, exclude_rated: bool = True, custom_alpha: float = None) -> list:
        """
        Generates Top-K hybrid recommendations for user_id.
        Excludes items already rated by user_id if exclude_rated is True.
        """
        alpha = custom_alpha if custom_alpha is not None else self.alpha
        rated_items = set()
        if user_id in self.ratings_df["user_id"].values and exclude_rated:
            rated_items = set(self.ratings_df[self.ratings_df["user_id"] == user_id]["item_id"])

        user_vec = self.cb_recommender.build_user_profile_vector(user_id)
        is_user_cold = user_vec is None

        # Precompute CB similarity scores for all items
        if not is_user_cold:
            cb_sims = cosine_similarity(user_vec, self.cb_recommender.pipeline.tfidf_matrix).flatten()
        else:
            cb_sims = np.zeros(len(self.items_df))

        # Check item rating counts to identify cold-start items
        item_rating_counts = self.ratings_df.groupby("item_id").size().to_dict()

        candidates = []
        for idx, item in self.items_df.iterrows():
            item_id = item["item_id"]
            if exclude_rated and item_id in rated_items:
                continue

            score_cb = float(cb_sims[idx]) if not is_user_cold else 0.5
            rating_count = item_rating_counts.get(item_id, 0)

            # Cold-start item adaptation
            if rating_count == 0 or is_user_cold:
                effective_alpha = 1.0 if not is_user_cold else 0.0
                score_cf = 0.0
                hybrid_score = score_cb if not is_user_cold else 0.5
            else:
                effective_alpha = alpha
                score_cf = self.cf_recommender.predict_rating(user_id, item_id)
                hybrid_score = effective_alpha * score_cb + (1.0 - effective_alpha) * score_cf

            candidates.append({
                "item_id": item_id,
                "grant_name": item["grant_name"],
                "country": item["country"],
                "degree_level": item["degree_level"],
                "score": round(hybrid_score, 4),
                "cb_score": round(score_cb, 4),
                "cf_score": round(score_cf, 4),
                "effective_alpha": round(effective_alpha, 2),
                "focus_tags": item["focus_tags"],
                "method": "Hybrid (Content-Based + Collaborative)",
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]


class ScholarshipRecommender:
    """Unified facade managing all three recommendation strategies."""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, "data")

        self.items_df = pd.read_csv(os.path.join(data_dir, "items.csv"))
        self.ratings_df = pd.read_csv(os.path.join(data_dir, "ratings.csv"))

        self.popularity = PopularityRecommender(self.ratings_df, self.items_df)
        self.content_based = ContentBasedRecommender(self.items_df, self.ratings_df)
        self.hybrid = HybridRecommender(self.ratings_df, self.items_df, alpha=0.5)

    def recommend_popularity(self, user_id: str = None, top_k: int = 5) -> list:
        return self.popularity.recommend(user_id=user_id, top_k=top_k)

    def recommend_content_based_query(self, query: str, top_k: int = 5) -> list:
        return self.content_based.recommend_for_query(query, top_k=top_k)

    def recommend_similar(self, item_id: str, top_k: int = 5) -> list:
        return self.content_based.recommend_similar_items(item_id, top_k=top_k)

    def recommend_content_based_user(self, user_id: str, top_k: int = 5) -> list:
        return self.content_based.recommend_for_user(user_id, top_k=top_k)

    def recommend_hybrid(self, user_id: str, top_k: int = 5, alpha: float = 0.5) -> list:
        return self.hybrid.recommend(user_id, top_k=top_k, custom_alpha=alpha)


if __name__ == "__main__":
    recommender = ScholarshipRecommender()
    print("--- 1. POPULARITY BASELINE (User U01) ---")
    for r in recommender.recommend_popularity(user_id="U01", top_k=5):
        print(f"[{r['item_id']}] {r['grant_name']}: Score={r['score']} (Avg={r['avg_rating']}, Count={r['rating_count']})")

    print("\n--- 2. CONTENT-BASED FOR USER U01 ---")
    for r in recommender.recommend_content_based_user(user_id="U01", top_k=5):
        print(f"[{r['item_id']}] {r['grant_name']}: Sim={r['score']}")

    print("\n--- 3. HYBRID FOR USER U01 ---")
    for r in recommender.recommend_hybrid(user_id="U01", top_k=5):
        print(f"[{r['item_id']}] {r['grant_name']}: Hybrid={r['score']} (CB={r['cb_score']}, CF={r['cf_score']})")
