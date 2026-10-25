
# Step 3: Import libraries and locate the recommender module

import sys
from pathlib import Path

import streamlit as st

# Find the main project folder
PROJECT_ROOT = Path(__file__).resolve().parent

# Find the src folder
SRC_DIR = PROJECT_ROOT / "src"

# Tell Python where to find recommender.py
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import the recommendation functions
from recommender import (
    load_restaurants,
    build_cuisine_features,
    recommend_restaurants
)


st.set_page_config(
    page_title="Restaurant Recommendation System",
    page_icon="🍽️",
    layout="wide"
)


@st.cache_data
def load_data():
    return load_restaurants()


@st.cache_resource
def load_features():
    restaurants = load_data()

    vectorizer, matrix = build_cuisine_features(
        restaurants
    )

    return vectorizer, matrix


st.title("🍽️ Restaurant Recommendation System")

st.write(
    "Discover restaurants based on your preferred cuisine, "
    "budget, rating, and location."
)

try:
    restaurants = load_data()
    vectorizer, cuisine_matrix = load_features()

except Exception as error:
    st.error(
        "The restaurant data could not be loaded. "
        "Check your dataset and column names."
    )

    st.exception(error)
    st.stop()


st.sidebar.header("Your Preferences")

cuisines = sorted({
    cuisine.strip()
    for description in restaurants["cuisines"]
    if description.strip()
    for cuisine in description.split(",")
    if cuisine.strip()
})

if not cuisines:
    st.error("No cuisine options are available.")
    st.stop()


preferred_cuisine = st.sidebar.selectbox(
    "Preferred cuisine",
    cuisines
)


valid_costs = restaurants[
    "average_cost_for_two"].dropna()

if not valid_costs.empty:
    maximum_available_cost = int(
        valid_costs.max()
    )

    maximum_budget = st.sidebar.number_input(
        "Maximum cost for two",
        min_value=0,
        max_value=max(1, maximum_available_cost),
        value=min(1500, max(1, maximum_available_cost)),
        step=100
    )

    apply_budget = st.sidebar.checkbox(
        "Apply maximum budget",
        value=True
    )

else:
    maximum_budget = None
    apply_budget = False

    st.sidebar.warning(
        "Cost information is unavailable."
    )


minimum_rating = st.sidebar.slider(
    "Minimum rating",
    min_value=0.0,
    max_value=5.0,
    value=3.0,
    step=0.5
)


selected_city = None

if "city" in restaurants.columns:
    cities = sorted({
        city.strip()
        for city in restaurants["city"]
        if city.strip()
    })

    if cities:
        selected_city = st.sidebar.selectbox(
            "Preferred city",
            ["All cities"] + cities
        )

        if selected_city == "All cities":
            selected_city = None


number_of_results = st.sidebar.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10
)


if st.button("Find Restaurants", type="primary"):

    try:
        results = recommend_restaurants(
            restaurants=restaurants,
            vectorizer=vectorizer,
            cuisine_matrix=cuisine_matrix,
            cuisine=preferred_cuisine,
            max_budget=(
                maximum_budget
                if apply_budget
                else None
            ),
            min_rating=minimum_rating,
            city=selected_city,
            top_n=number_of_results
        )

        if results.empty:
            st.warning(
                "No restaurants match all your preferences. "
                "Try another cuisine, reduce the minimum "
                "rating, or increase your budget."
            )

        else:
            st.success(
                f"Found {len(results)} restaurant recommendations."
            )

            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True
            )

            csv_data = results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download recommendations as CSV",
                data=csv_data,
                file_name="restaurant_recommendations.csv",
                mime="text/csv"
            )

    except Exception as error:
        st.error(
            "Unable to generate recommendations. "
            "Please check your preferences and dataset."
        )

        st.exception(error)


st.divider()

st.caption(
    "Cognifyz Technology | Task 2: Restaurant Recommendation System"
)
