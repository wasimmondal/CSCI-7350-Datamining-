"""
recommendationA.py

Modified user-based collaborative filtering for MovieLens 100K.

Main modifications over the baseline:
1. Uses Pearson correlation with a minimum number of shared movies.
2. Uses only the Top-K most similar neighbors.
3. Uses mean-centered prediction with user-average fallback.
4. Evaluates on 200 randomly selected test users with random.seed(42).

Expected input file:
    movies100K/u.data

Run:
    python recommendationA.py

Optional:
    python recommendationA.py --data movies100K/u.data --k 20 --min-overlap 5
"""

import argparse
import math
import random
from collections import defaultdict


def load_ratings(file_path):
    """
    Load MovieLens ratings from u.data format:
    user_id \t movie_id \t rating \t timestamp
    """
    user_ratings = defaultdict(dict)

    with open(file_path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 4:
                continue
            user_id, movie_id, rating, _ = parts
            user_ratings[user_id][movie_id] = float(rating)

    return dict(user_ratings)


def user_mean(ratings_dict):
    """Return average rating for one user."""
    if not ratings_dict:
        return 3.0
    return sum(ratings_dict.values()) / len(ratings_dict)


def pearson_similarity_with_overlap(user1_ratings, user2_ratings, min_overlap=5):
    """
    Pearson similarity between two users with a minimum shared-movie threshold.
    Returns 0 if overlap is too small or denominator is zero.
    """
    shared_items = set(user1_ratings).intersection(user2_ratings)
    n = len(shared_items)

    if n < min_overlap:
        return 0.0

    mean1 = sum(user1_ratings[m] for m in shared_items) / n
    mean2 = sum(user2_ratings[m] for m in shared_items) / n

    num = sum(
        (user1_ratings[m] - mean1) * (user2_ratings[m] - mean2)
        for m in shared_items
    )
    den1 = math.sqrt(sum((user1_ratings[m] - mean1) ** 2 for m in shared_items))
    den2 = math.sqrt(sum((user2_ratings[m] - mean2) ** 2 for m in shared_items))

    if den1 == 0 or den2 == 0:
        return 0.0

    return num / (den1 * den2)


def predict_rating_topk(
    train_data,
    target_user,
    target_movie,
    k=20,
    min_overlap=5,
):
    """
    Predict one missing rating for target_user on target_movie using:
    - user-based collaborative filtering
    - Pearson similarity
    - minimum overlap threshold
    - Top-K neighbors only
    - mean-centered weighted prediction

    Returns None if prediction is not possible.
    """
    if target_user not in train_data:
        return None

    target_user_ratings = train_data[target_user]
    target_user_mean = user_mean(target_user_ratings)

    neighbors = []

    for other_user, other_ratings in train_data.items():
        if other_user == target_user:
            continue

        if target_movie not in other_ratings:
            continue

        sim = pearson_similarity_with_overlap(
            target_user_ratings,
            other_ratings,
            min_overlap=min_overlap,
        )

        if sim > 0:
            neighbors.append((sim, other_user))

    if not neighbors:
        return target_user_mean

    neighbors.sort(reverse=True, key=lambda x: x[0])
    top_neighbors = neighbors[:k]

    numerator = 0.0
    denominator = 0.0

    for sim, neighbor in top_neighbors:
        neighbor_mean = user_mean(train_data[neighbor])
        neighbor_rating = train_data[neighbor][target_movie]

        numerator += sim * (neighbor_rating - neighbor_mean)
        denominator += abs(sim)

    if denominator == 0:
        return target_user_mean

    pred = target_user_mean + (numerator / denominator)

    # Clip to MovieLens rating range
    pred = max(1.0, min(5.0, pred))
    return pred


def pearson_correlation(xs, ys):
    """
    Pearson correlation between two equal-length lists.
    Returns 0 if not enough data or zero variance.
    """
    n = len(xs)
    if n < 2:
        return 0.0

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))

    if den_x == 0 or den_y == 0:
        return 0.0

    return num / (den_x * den_y)


def evaluate_system(all_data, num_test_users=200, random_seed=42, k=20, min_overlap=5):
    """
    Evaluation procedure:
    - Select 200 random users for testing
    - For each test user, hide one rated movie at a time
    - Predict that hidden rating using:
        training users + target user's remaining ratings
    - Compute Pearson correlation between actual and predicted ratings for each user
    - Return average user correlation
    """
    random.seed(random_seed)

    eligible_users = [u for u, ratings in all_data.items() if len(ratings) >= 2]

    if len(eligible_users) < num_test_users:
        test_users = eligible_users
    else:
        test_users = random.sample(eligible_users, num_test_users)

    user_correlations = []

    for user in test_users:
        actuals = []
        preds = []

        user_movies = list(all_data[user].keys())

        for hidden_movie in user_movies:
            # Build train data excluding the full target user first
            train_data = {}

            for other_user, ratings in all_data.items():
                if other_user != user:
                    train_data[other_user] = dict(ratings)

            # Add target user back with one rating hidden
            partial_profile = dict(all_data[user])
            true_rating = partial_profile.pop(hidden_movie)

            if not partial_profile:
                continue

            train_data[user] = partial_profile

            pred = predict_rating_topk(
                train_data=train_data,
                target_user=user,
                target_movie=hidden_movie,
                k=k,
                min_overlap=min_overlap,
            )

            if pred is None:
                continue

            preds.append(pred)
            actuals.append(true_rating)

        if len(actuals) >= 2:
            corr = pearson_correlation(actuals, preds)
            user_correlations.append(corr)

    avg_corr = sum(user_correlations) / len(user_correlations) if user_correlations else 0.0

    return avg_corr, user_correlations, test_users


def main():
    parser = argparse.ArgumentParser(description="RecommendationA: improved user-based CF")
    parser.add_argument(
        "--data",
        type=str,
        default="movies100K/u.data",
        help="Path to MovieLens u.data file",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Number of top neighbors to use",
    )
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=5,
        help="Minimum number of shared movies for similarity",
    )
    parser.add_argument(
        "--num-test-users",
        type=int,
        default=200,
        help="Number of random test users",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    all_data = load_ratings(args.data)

    avg_corr, user_corrs, test_users = evaluate_system(
        all_data=all_data,
        num_test_users=args.num_test_users,
        random_seed=args.seed,
        k=args.k,
        min_overlap=args.min_overlap,
    )

    print("=== RecommendationA Results ===")
    print(f"Data file               : {args.data}")
    print(f"Random seed             : {args.seed}")
    print(f"Number of test users    : {len(test_users)}")
    print(f"Top-K neighbors         : {args.k}")
    print(f"Minimum shared movies   : {args.min_overlap}")
    print(f"Users with valid corr   : {len(user_corrs)}")
    print(f"Average Pearson corr    : {avg_corr:.4f}")


if __name__ == "__main__":
    main()
