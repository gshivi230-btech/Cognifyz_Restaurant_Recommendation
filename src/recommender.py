
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Dataset .csv"
if not DATA_PATH.exists() and (PROJECT_ROOT / "data" / "Dataset.csv").exists():
    DATA_PATH = PROJECT_ROOT / "data" / "Dataset.csv"


def normalize_column_name(column):
    """Normalize column names for consistent processing."""
    import re

    name = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        str(column).strip().lower()
    )

    return name.strip("_")


def find_column(columns, candidates):
    """Find a column using common alternative names."""
    for candidate in candidates:
        if candidate in columns:
            return candidate

    return None


def load_restaurants():
    """Load and standardize restaurant data."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Restaurant dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    df.columns = [
        normalize_column_name(column)
        for column in df.columns
    ]

    if df.columns.duplicated().any():
        raise ValueError(
            "Multiple dataset columns have identical names "
            "after normalization. Please resolve them."
        )

    columns = set(df.columns)

    column_map = {
        "name": find_column(
            columns,
            ["restaurant_name", "name", "restaurant"]
        ),
        "cuisine": find_column(
            columns,
            ["cuisines", "cuisine", "cuisine_type"]
        ),
        "rating": find_column(
            columns,
            ["aggregate_rating", "rating", "rate"]
        ),
        "cost": find_column(
            columns,
            [
                "average_cost_for_two",
                "average_cost",
                "cost_for_two"
            ]
        ),
        "city": find_column(
            columns,
            ["city", "city_name"]
        )
    }

    missing = [
        key
        for key in ["name", "cuisine", "rating", "cost"]
        if column_map[key] is None
    ]

    if missing:
        raise ValueError(
            f"Could not identify required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    restaurants = pd.DataFrame({
        "restaurant_name": (
            df[column_map["name"]]
            .astype("string")
            .str.strip()
        ),
        "cuisines": (
            df[column_map["cuisine"]]
            .fillna("")
            .astype(str)
            .str.strip()
        ),
        "aggregate_rating": pd.to_numeric(
            df[column_map["rating"]]
            .replace("", pd.NA),
            errors="coerce"
        ),
        "average_cost_for_two": pd.to_numeric(
            df[column_map["cost"]]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
            .replace("", pd.NA),
            errors="coerce"
        )
    })

    if column_map["city"] is not None:
        restaurants["city"] = (
            df[column_map["city"]]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    restaurants = restaurants.drop_duplicates().copy()

    restaurants = restaurants[
        restaurants["restaurant_name"].notna()
        & restaurants["restaurant_name"].ne("")
    ].copy()

    restaurants.reset_index(drop=True, inplace=True)

    # Ignore impossible values rather than recommending them.
    restaurants.loc[
        ~restaurants["aggregate_rating"].between(0, 5),
        "aggregate_rating"
    ] = float("nan")

    restaurants.loc[
        restaurants["average_cost_for_two"] < 0,
        "average_cost_for_two"
    ] = float("nan")

    if not restaurants["cuisines"].str.strip().ne("").any():
        raise ValueError(
            "No usable cuisine descriptions were found."
        )

    return restaurants


def build_cuisine_features(restaurants):
    """Build a TF-IDF representation of cuisine descriptions."""

    vectorizer = TfidfVectorizer(
        tokenizer=lambda text: [
            item.strip().lower()
            for item in text.split(",")
            if item.strip()
        ],
        token_pattern=None,
        lowercase=False
    )

    matrix = vectorizer.fit_transform(
        restaurants["cuisines"]
    )

    return vectorizer, matrix


def recommend_restaurants(
    restaurants,
    vectorizer,
    cuisine_matrix,
    cuisine,
    max_budget=None,
    min_rating=0,
    city=None,
    top_n=10
):
    """Recommend restaurants using cuisine similarity."""

    cuisine = str(cuisine).strip()

    if not cuisine:
        raise ValueError("Please enter a cuisine preference.")

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    if max_budget is not None and max_budget < 0:
        raise ValueError("Budget cannot be negative.")

    if not 0 <= min_rating <= 5:
        raise ValueError("Rating must be between 0 and 5.")

    user_vector = vectorizer.transform([cuisine])

    scores = cosine_similarity(
        user_vector,
        cuisine_matrix
    ).flatten()

    results = restaurants.copy()
    results["cuisine_similarity"] = scores

    results = results[
        results["cuisines"].str.strip().ne("")
        & (results["cuisine_similarity"] > 0)
    ].copy()

    if max_budget is not None:
        results = results[
            results["average_cost_for_two"].notna()
            & (
                results["average_cost_for_two"]
                <= max_budget
            )
        ]

    if min_rating > 0:
        results = results[
            results["aggregate_rating"].notna()
            & (
                results["aggregate_rating"]
                >= min_rating
            )
        ]

    if city:
        if "city" not in results.columns:
            raise ValueError(
                "City filtering is unavailable in this dataset."
            )

        results = results[
            results["city"].str.casefold()
            == city.strip().casefold()
        ]

    results = results.sort_values(
        by=["cuisine_similarity", "aggregate_rating"],
        ascending=[False, False],
        na_position="last"
    )

    return results.head(top_n).reset_index(drop=True)
