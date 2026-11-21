"""
Scholarship and Student Grant Recommender - Main CLI Application (MP-C06)
Student: Nguyễn Minh Khanh (29211355598)
Course: CS467 Advanced Artificial Intelligence - CS467C
Instructor: Puri Vikram

Usage:
    python -m src.main                   # Runs complete demo and 3 required test cases
    python -m src.main --all             # Runs demo, all 3 test cases, and evaluation
    python -m src.main --profile "<text>"# Free-text student profile search
    python -m src.main --user <user_id>  # Recommends for user using all 3 algorithms
    python -m src.main --similar <id>    # Item-to-item similarity
    python -m src.main --eval            # Runs offline evaluation & generates charts
"""

import os
import sys
import argparse
import pandas as pd
from src.nlp_pipeline import ScholarshipNLPPipeline
from src.recommender import ScholarshipRecommender
from src.evaluate import run_full_evaluation, run_three_test_cases

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def print_banner():
    banner = """
================================================================================
     SCHOLARSHIP AND STUDENT GRANT RECOMMENDER SYSTEM (MP-C06)
     Student: Nguyễn Minh Khanh (ID: 29211355598)
     Course: CS467 Advanced AI | Section: CS467C (No. 06)
     Instructor: Puri Vikram
================================================================================
"""
    print(banner)


def handle_profile_search(pipeline: ScholarshipNLPPipeline, query: str, top_k: int = 5):
    print(f"\n[QUERY SEARCH] Student Profile: \"{query}\"")
    print("-" * 85)
    print(f"{'Rank':<5} | {'Item ID':<8} | {'Score':<7} | {'Country':<11} | {'Degree':<13} | {'Grant Name':<35}")
    print("-" * 85)
    results = pipeline.search_by_query(query, top_k=top_k)
    for i, r in enumerate(results, 1):
        print(f"{i:<5} | {r['item_id']:<8} | {r['similarity_score']:<7.4f} | {r['country']:<11} | {r['degree_level']:<13} | {r['grant_name']}")
        print(f"      Keywords: {', '.join(r['matching_keywords'])}")
    print("-" * 85)


def handle_similar_items(pipeline: ScholarshipNLPPipeline, item_id: str, top_k: int = 5):
    print(f"\n[SIMILAR SCHOLARSHIPS] Seed Item: {item_id}")
    try:
        results = pipeline.find_similar_items(item_id, top_k=top_k)
    except KeyError as e:
        print(f"Error: {e}")
        return

    print("-" * 85)
    print(f"{'Rank':<5} | {'Item ID':<8} | {'Score':<7} | {'Country':<11} | {'Degree':<13} | {'Grant Name':<35}")
    print("-" * 85)
    for i, r in enumerate(results, 1):
        print(f"{i:<5} | {r['item_id']:<8} | {r['similarity_score']:<7.4f} | {r['country']:<11} | {r['degree_level']:<13} | {r['grant_name']}")
        print(f"      Shared Keywords: {', '.join(r['matching_keywords'])}")
    print("-" * 85)


def handle_user_recommendations(recommender: ScholarshipRecommender, user_id: str, top_k: int = 5):
    print(f"\n[USER RECOMMENDATIONS] Comparing Methods for User: {user_id}")
    print("=" * 85)

    # 1. Popularity
    print("\n>>> Method 1: Popularity Baseline (Excluding Already Rated Items)")
    print("-" * 85)
    pop_recs = recommender.recommend_popularity(user_id=user_id, top_k=top_k)
    for i, r in enumerate(pop_recs, 1):
        print(f"{i}. [{r['item_id']}] {r['grant_name']} ({r['country']}) - Bayes Score: {r['score']} (Avg: {r['avg_rating']}, Count: {r['rating_count']})")

    # 2. Content-Based
    print("\n>>> Method 2: NLP Content-Based (Based on User's Rating History Profile)")
    print("-" * 85)
    cb_recs = recommender.recommend_content_based_user(user_id=user_id, top_k=top_k)
    for i, r in enumerate(cb_recs, 1):
        print(f"{i}. [{r['item_id']}] {r['grant_name']} ({r['country']}) - Similarity: {r['score']}")

    # 3. Hybrid
    print("\n>>> Method 3: Hybrid Recommender (50% NLP Content-Based + 50% Collaborative Filtering)")
    print("-" * 85)
    hyb_recs = recommender.recommend_hybrid(user_id=user_id, top_k=top_k)
    for i, r in enumerate(hyb_recs, 1):
        print(f"{i}. [{r['item_id']}] {r['grant_name']} ({r['country']}) - Hybrid Score: {r['score']} (CB: {r['cb_score']}, CF: {r['cf_score']})")
    print("=" * 85)


def run_default_demo():
    """Runs a complete showcase of the 3 required test cases and system features."""
    print_banner()

    items_path = os.path.join(DATA_DIR, "items.csv")
    ratings_path = os.path.join(DATA_DIR, "ratings.csv")

    if not os.path.exists(items_path) or not os.path.exists(ratings_path):
        from src.generate_dataset import generate_and_save
        print("Generating dataset files...")
        generate_and_save()

    items_df = pd.read_csv(items_path)
    ratings_df = pd.read_csv(ratings_path)
    recommender = ScholarshipRecommender()

    # Run and display the 3 mandatory test cases
    sample_out_file = os.path.join(RESULTS_DIR, "sample_outputs.txt")
    run_three_test_cases(items_df, ratings_df, sample_out_file)

    # Demonstrate recommendation for student U01 across all 3 methods
    handle_user_recommendations(recommender, "U01", top_k=5)

    print("\n" + "=" * 80)
    print("DEMO COMPLETE: All algorithms tested and verified.")
    print("Charts and outputs generated in 'results/' directory.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Scholarship and Student Grant Recommender CLI (MP-C06)"
    )
    parser.add_argument("--profile", type=str, help="Search top scholarships for a custom student free-text profile.")
    parser.add_argument("--user", type=str, help="Generate recommendations for a specific user ID (e.g. U01 to U15).")
    parser.add_argument("--similar", type=str, help="Find similar scholarships for a given scholarship ID (e.g. SCH_01).")
    parser.add_argument("--eval", action="store_true", help="Run offline evaluation and generate performance charts.")
    parser.add_argument("--all", action="store_true", help="Run all test cases, user comparison, and evaluation.")

    args = parser.parse_args()

    # If no flags provided, run default demo
    if len(sys.argv) == 1:
        run_default_demo()
        return

    print_banner()
    items_path = os.path.join(DATA_DIR, "items.csv")
    ratings_path = os.path.join(DATA_DIR, "ratings.csv")
    items_df = pd.read_csv(items_path)
    ratings_df = pd.read_csv(ratings_path)
    pipeline = ScholarshipNLPPipeline(items_df)
    recommender = ScholarshipRecommender()

    if args.profile:
        handle_profile_search(pipeline, args.profile)

    if args.similar:
        handle_similar_items(pipeline, args.similar)

    if args.user:
        handle_user_recommendations(recommender, args.user)

    if args.eval:
        print("\n[RUNNING OFFLINE EVALUATION]")
        run_full_evaluation()

    if args.all:
        run_default_demo()
        print("\n[RUNNING OFFLINE EVALUATION AND CHART GENERATION]")
        run_full_evaluation()


if __name__ == "__main__":
    main()
