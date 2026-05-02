"""
recommendationB.py

Item-based collaborative filtering for MovieLens 100K.

Method:
1. Compute similarity between movies instead of users.
2. Predict a user's rating for a target movie using the user's own ratings
   on similar movies.
3. Use adjusted cosine similarity (mean-centered by user averages).
4. Use Top-K similar items for prediction.
5. Evaluate on 200 randomly selected test users with random.seed(42).

Expected input file:
    movies100K/u.data

Run:
    python recommendationB.py

Optional:
    python recommendationB.py --data movies100K/u.data --k 20 --min-overlap 5
"""

import argparse
import math
import random
from collections import defaultdict


def load_ratings(file_path):
    """
    Load MovieLens ratings from u.data format:
    user_id \t movie_id \t rating \t timestamp
    Returns:
        user_ratings: {user: {movie: rating}}
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


def build_item_users(user_ratings):
    """
    Convert user-based dictionary to item-based dictionary:
        {movie: {user: rating}}
    """
    item_users = defaultdict(dict)
    for user, ratings in user_ratings.items():
        for movie, rating in ratings.items():
            item_users[movie][user] = rating
    return dict(item_users)


def compute_user_means(user_ratings):
    """
    Precompute average rating for each user.
    """
    means = {}
    for user, ratings in user_ratings.items():
        if ratings:
            means[user] = sum(ratings.values()) / len(ratings)
        else:
            means[user] = 3.0
    return means


def adjusted_cosine_similarity(item1_users, item2_users, user_means, min_overlap=5):
    """
    Adjusted cosine similarity between two movies.

    Ratings are centered by each user's mean rating.
    Only users who rated both movies are used.
    """
    shared_users = set(item1_users).intersection(item2_users)
    if len(shared_users) < min_overlap:
        return 0.0

    numerator = 0.0
    denom1 = 0.0
    denom2 = 0.0

    for user in shared_users:
        r1 = item1_users[user] - user_means[user]
        r2 = item2_users[user] - user_means[user]

        numerator += r1 * r2
        denom1 += r1 * r1
        denom2 += r2 * r2

    if denom1 == 0 or denom2 == 0:
        return 0.0

    return numerator / math.sqrt(denom1 * denom2)


def predict_rating_item_based(
    train_data,
    target_user,
    target_movie,
    k=20,
    min_overlap=5,
    similarity_cache=None,
):
    """
    Predict a user's rating for a target movie using item-based CF.

    Steps:
    - Find all movies already rated by the target user
    - Compute similarity between target movie and each rated movie
    - Keep only positive similarities
    - Use Top-K most similar items
    - Weighted mean-centered prediction

    Returns:
        predicted rating (float) or None
    """
    if target_user not in train_data:
        return None

    target_user_ratings = train_data[target_user]
    if not target_user_ratings:
        return None

    user_means = compute_user_means(train_data)
    item_users = build_item_users(train_data)

    target_user_mean = user_means[target_user]

    if target_movie not in item_users:
        return target_user_mean

    neighbors = []

    for other_movie, rating in target_user_ratings.items():
        if other_movie == target_movie:
            continue

        key = tuple(sorted((target_movie, other_movie)))
        if similarity_cache is not None and key in similarity_cache:
            sim = similarity_cache[key]
        else:
            if other_movie not in item_users:
                sim = 0.0
            else:
                sim = adjusted_cosine_similarity(
                    item_users[target_movie],
                    item_users[other_movie],
                    user_means,
                    min_overlap=min_overlap,
                )
            if similarity_cache is not None:
                similarity_cache[key] = sim

        if sim > 0:
            neighbors.append((sim, other_movie, rating))

    if not neighbors:
        return target_user_mean

    neighbors.sort(reverse=True, key=lambda x: x[0])
    top_neighbors = neighbors[:k]

    numerator = 0.0
    denominator = 0.0

    for sim, other_movie, rating in top_neighbors:
        numerator += sim * (rating - target_user_mean)
        denominator += abs(sim)

    if denominator == 0:
        return target_user_mean

    pred = target_user_mean + (numerator / denominator)
    pred = max(1.0, min(5.0, pred))
    return pred


def pearson_correlation(xs, ys):
    """
    Pearson correlation between two equal-length lists.
    """
    n = len(xs)
    if n < 2:
        return 0.0

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))

    if denom_x == 0 or denom_y == 0:
        return 0.0

    return numerator / (denom_x * denom_y)


def evaluate_system(all_data, num_test_users=200, random_seed=42, k=20, min_overlap=5):
    """
    Evaluation procedure:
    - Randomly select 200 users for testing
    - For each test user, hide one rating at a time
    - Predict the hidden rating using training data plus remaining ratings
    - Compute Pearson correlation between predicted and actual ratings
    - Average user-level correlations
    """
    random.seed(random_seed)

    eligible_users = [u for u, ratings in all_data.items() if len(ratings) >= 2]
    test_users = random.sample(eligible_users, min(num_test_users, len(eligible_users)))

    user_correlations = []
    similarity_cache = {}

    for user in test_users:
        actuals = []
        preds = []

        user_movies = list(all_data[user].keys())

        for hidden_movie in user_movies:
            train_data = {}

            for other_user, ratings in all_data.items():
                if other_user != user:
                    train_data[other_user] = dict(ratings)

            partial_profile = dict(all_data[user])
            true_rating = partial_profile.pop(hidden_movie)

            if not partial_profile:
                continue

            train_data[user] = partial_profile

            pred = predict_rating_item_based(
                train_data=train_data,
                target_user=user,
                target_movie=hidden_movie,
                k=k,
                min_overlap=min_overlap,
                similarity_cache=similarity_cache,
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
    parser = argparse.ArgumentParser(description="RecommendationB: item-based CF")
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
        help="Number of top similar items to use",
    )
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=5,
        help="Minimum common users required for item similarity",
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

    print("=== RecommendationB Results ===")
    print(f"Data file               : {args.data}")
    print(f"Random seed             : {args.seed}")
    print(f"Number of test users    : {len(test_users)}")
    print(f"Top-K similar items     : {args.k}")
    print(f"Minimum shared users    : {args.min_overlap}")
    print(f"Users with valid corr   : {len(user_corrs)}")
    print(f"Average Pearson corr    : {avg_corr:.4f}")


if __name__ == "__main__":
    main()
