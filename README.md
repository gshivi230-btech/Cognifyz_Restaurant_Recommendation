# 🍽️ Cognifyz Restaurant Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cognifyz-restaurant-recommender.streamlit.app/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Online-brightgreen.svg?logo=streamlit&logoColor=white)](https://cognifyz-restaurant-recommender.streamlit.app/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Cognifyz Technologies Data Science Internship — Task 2: Restaurant Recommendation System**  
> An intelligent content-based recommendation engine built with Python, TF-IDF vectorization, cosine similarity, and an interactive Streamlit web dashboard.

---

## 🌐 Live Demo

The application is deployed and live on **Streamlit Community Cloud**:  
👉 **[Open Live App: cognifyz-restaurant-recommender.streamlit.app](https://cognifyz-restaurant-recommender.streamlit.app/)**

---

## 📌 Project Overview

This project develops a content-based restaurant recommendation system that suggests restaurants based on user preferences including **cuisine type**, **budget (average cost for two)**, **minimum aggregate rating**, and **city location**.

The recommendation algorithm utilizes **Term Frequency-Inverse Document Frequency (TF-IDF)** to extract cuisine feature vectors and computes **Cosine Similarity** between user preferences and restaurant profiles to rank the top matching venues.

---

## ✨ Features

- **Content-Based Filtering**: TF-IDF vectorization and cosine similarity scoring for cuisine matching.
- **Multi-Criteria Filtering**:
  - 🍕 **Preferred Cuisine**: Dynamic dropdown populated from available cuisine descriptions.
  - 💵 **Maximum Budget**: Customizable threshold for average cost for two.
  - ⭐ **Minimum Rating**: Slider-based rating filter (0.0 to 5.0).
  - 📍 **City Filter**: Location-based filtering with support for all cities.
- **Top-N Customization**: Select between 5 and 20 recommendations.
- **Interactive UI**: Clean, responsive web dashboard built with Streamlit.
- **CSV Export**: One-click download button for recommended results.
- **Robust Error Handling**: Graceful fallback for missing fields, inconsistent naming, and out-of-bounds ratings.

---

## 📁 Repository Structure

```plaintext
Cognifyz_Restaurant_Recommendation/
├── app.py                     # Streamlit web application frontend
├── requirements.txt           # Python package dependencies
├── .gitignore                 # Files and folders ignored by Git
├── README.md                  # Project documentation
├── data/
│   └── Dataset .csv           # Cleaned restaurant dataset
├── src/
│   └── recommender.py         # Recommendation algorithms & data loading
├── outputs/
│   ├── recommendation_evaluation.csv
│   └── restaurant_recommendations.csv
└── notebook/
    └── restaurant_recommendation.ipynb  # Exploratory Data Analysis & experiments
```

---

## 🛠️ Tech Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/)
- **Machine Learning & NLP**: [Scikit-learn](https://scikit-learn.org/) (`TfidfVectorizer`, `cosine_similarity`)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)

---

## 🚀 Local Installation & Setup

Follow these steps to run the application locally on your machine:

### 1. Clone the Repository
```bash
git clone https://github.com/gshivi230-btech/Cognifyz_Restaurant_Recommendation.git
cd Cognifyz_Restaurant_Recommendation
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## ☁️ Deployment on Streamlit Community Cloud

The application is deployed on **Streamlit Community Cloud**:

- 🌐 **Live Web App**: [https://cognifyz-restaurant-recommender.streamlit.app/](https://cognifyz-restaurant-recommender.streamlit.app/)
- **Repository**: `gshivi230-btech/Cognifyz_Restaurant_Recommendation`
- **Branch**: `main`
- **Main file path**: `app.py`

Any changes committed and pushed to the `main` branch will automatically trigger a re-deployment on Streamlit Community Cloud.

---

## 📊 Evaluation & Results

The system evaluates recommendation quality using similarity thresholds and rating constraints, ensuring users receive relevant, high-rated, and budget-friendly suggestions.

---

## 👤 Author

- **Medhavie Gupta** ([@gshivi230-btech](https://github.com/gshivi230-btech))
- **Program**: Cognifyz Technologies Internship — Task 2

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
