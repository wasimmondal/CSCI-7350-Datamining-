"""
baseline_from_notebook.py

Baseline user-based collaborative filtering, adapted from recommendationDemo.ipynb
and extended to evaluate the average Pearson correlation on MovieLens 100K test users.

What this script does:
1. Loads ratings from movies100K/u.data
2. Uses user-based collaborative filtering
3. Uses Pearson correlation for user similarity
4. Uses all users with positive similarity (baseline behavior)
5. Randomly selects 200 test users with random.seed(42)
6. Hides one rating at a time for each test user
7. Computes the Pearson correlation between predicted and actual ratings per user
8. Reports the average correlation across test users

Run:
    python3.11 baseline_from_notebook.py

Optional:
    python3.11 baseline_from_notebook.py --data movies100K/u.data --num-test-users 200 --seed 42
"""

import argparse
import math
import random
from collections import defaultdict


def load_ratings(file_path):
    """Load MovieLens ratings from u.data format."""
    prefs = defaultdict(dict)
    with open(file_path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 4:
                continue
            user_id, movie_id, rating, _ = parts
            prefs[user_id][movie_id] = float(rating)
    return dict(prefs)


def sim_pearson(prefs, person1, person2):
    """
    Pearson correlation coefficient between two users.
    This follows the notebook's baseline idea, but without debug printing.
    """
    shared = [item for item in prefs[person1] if item in prefs[person2]]
    n = len(shared)

    if n == 0:
        return 0.0

    sum1 = sum(prefs[person1][it] for it in shared)
    sum2 = sum(prefs[person2][it] for it in shared)

    sum1_sq = sum(prefs[person1][it] ** 2 for it in shared)
    sum2_sq = sum(prefs[person2][it] ** 2 for it in shared)

    p_sum = sum(prefs[person1][it] * prefs[person2][it] for it in shared)

    num = p_sum - (sum1 * sum2 / n)
    den = math.sqrt((sum1_sq - (sum1 ** 2) / n) * (sum2_sq - (sum2 ** 2) / n))

    if den == 0:
        return 0.0

    return num / den


def user_mean(ratings_dict):
    """Average rating of a user."""
    if not ratings_dict:
        return 3.0
    return sum(ratings_dict.values()) / len(ratings_dict)


def predict_rating_baseline(prefs, person, target_item):
    """
    Baseline prediction using all users with positive Pearson similarity.
    Weighted average of neighbors' ratings, matching the project baseline description.
    """
    totals = 0.0
    sim_sums = 0.0

    for other in prefs:
        if other == person:
            continue
        if target_item not in prefs[other]:
            continue

        sim = sim_pearson(prefs, person, other)

        # Baseline uses all users with positive similarity
        if sim <= 0:
            continue

        totals += prefs[other][target_item] * sim
        sim_sums += sim

    if sim_sums == 0:
        # reasonable fallback if no positive-similarity neighbor is found
        return user_mean(prefs[person])

    pred = totals / sim_sums
    return max(1.0, min(5.0, pred))


def pearson_correlation(xs, ys):
    """Pearson correlation between two lists."""
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


def evaluate_baseline(all_data, num_test_users=200, random_seed=42):
    """
    Project-style evaluation:
    - Randomly choose test users
    - For each test user, hide one rating at a time
    - Predict with the baseline system using the user's remaining ratings
    - Compute Pearson correlation between predicted and actual ratings for each user
    - Return average user correlation
    """
    random.seed(random_seed)

    eligible_users = [u for u, ratings in all_data.items() if len(ratings) >= 2]
    test_users = random.sample(eligible_users, min(num_test_users, len(eligible_users)))

    user_correlations = []

    for user in test_users:
        actuals = []
        preds = []

        for hidden_item in list(all_data[user].keys()):
            # training data = all other users + target user's remaining ratings
            train_data = {other: dict(ratings) for other, ratings in all_data.items() if other != user}

            partial_profile = dict(all_data[user])
            true_rating = partial_profile.pop(hidden_item)

            if not partial_profile:
                continue

            train_data[user] = partial_profile

            pred = predict_rating_baseline(train_data, user, hidden_item)
            preds.append(pred)
            actuals.append(true_rating)

        if len(actuals) >= 2:
            corr = pearson_correlation(actuals, preds)
            user_correlations.append(corr)

    avg_corr = sum(user_correlations) / len(user_correlations) if user_correlations else 0.0
    return avg_corr, user_correlations, test_users


def main():
    parser = argparse.ArgumentParser(description="Baseline CF adapted from recommendationDemo.ipynb")
    parser.add_argument("--data", type=str, default="movies100K/u.data", help="Path to MovieLens u.data")
    parser.add_argument("--num-test-users", type=int, default=200, help="Number of random test users")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    all_data = load_ratings(args.data)
    avg_corr, user_corrs, test_users = evaluate_baseline(
        all_data=all_data,
        num_test_users=args.num_test_users,
        random_seed=args.seed,
    )

    print("=== Baseline Results ===")
    print(f"Data file               : {args.data}")
    print(f"Random seed             : {args.seed}")
    print(f"Number of test users    : {len(test_users)}")
    print(f"Users with valid corr   : {len(user_corrs)}")
    print(f"Average Pearson corr    : {avg_corr:.4f}")


if __name__ == "__main__":
    main()
