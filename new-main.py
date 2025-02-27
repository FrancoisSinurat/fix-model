from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import re
import string
from langdetect import detect
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error

app = Flask(__name__)
CORS(app)

# ============================================= LOAD DATASET =============================================
courses_df = pd.read_csv("dataset-fix-SinityCourse/Coursera_courses.csv")
reviews_df = pd.read_csv("dataset-fix-SinityCourse/Coursera_reviews.csv")

# Clean Text Function
def clean_text(text):
    try:
        if not text or detect(text) != 'en':
            return ""
    except:
        return ""
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    selected_sentences = " ".join(sentences[:3])

    selected_sentences = re.sub(r"[^a-zA-Z0-9\s.!?]", "", selected_sentences)
    selected_sentences = re.sub(r"\s+", " ", selected_sentences).strip()
    
    return selected_sentences

reviews_df['reviews'] = reviews_df['reviews'].astype(str).apply(clean_text)

# ============================================= PREPROCESSING COURSE =============================================
courses_df.drop(columns=['institution', 'course_url'], inplace=True)

label_encoder = LabelEncoder()
courses_df['course_id_unq'] = label_encoder.fit_transform(courses_df['course_id'])

course_id_mapping = pd.DataFrame({
    'course_id': courses_df['course_id'],
    'course_id_unq': courses_df['course_id_unq']
})

# ============================================= PREPROCESSING REVIEWS =============================================
reviews_df.drop(columns=['date_reviews'], inplace=True)
reviews_df = reviews_df.merge(course_id_mapping, on='course_id', how='left')

user_counts = reviews_df['reviewers'].value_counts()
multiple_ratings = user_counts[user_counts >= 2].index
user_id_mapping = {reviewer: idx + 1 for idx, reviewer in enumerate(multiple_ratings)}

reviews_filtered = reviews_df[reviews_df['reviewers'].isin(multiple_ratings)]
reviews_filtered1 = reviews_filtered.drop_duplicates(subset=['reviewers', 'course_id'])

user_id_mapping = {}
current_id = 1
for reviewer in reviews_df['reviewers']:
    if reviewer not in user_id_mapping:
        user_id_mapping[reviewer] = current_id
        current_id += 1

reviews_filtered1 = reviews_filtered1.copy()
reviews_filtered1['user_id'] = reviews_filtered1['reviewers'].map(user_id_mapping)

# ============================================= TRAIN-TEST SPLIT =============================================
train_data, test_data = train_test_split(reviews_filtered1, test_size=0.3, random_state=42)
train_data.to_csv("train_reviews.csv", index=False)
test_data.to_csv("test_reviews.csv", index=False)

# ============================================= CONTENT-BASED FILTERING =============================================
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(courses_df["description"])
cosine_sim = cosine_similarity(tfidf_matrix)

def recommend_courses_content(course_id, top_n=5):
    idx = courses_df[courses_df["course_id"] == course_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    course_indices = [i[0] for i in sim_scores]
    return courses_df.iloc[course_indices][["course_id", "title"]].to_dict(orient="records")

# ============================================= COLLABORATIVE FILTERING (SVD) =============================================
user_course_matrix = train_data.pivot(index="user_id", columns="course_id_unq", values="ratings").fillna(0)

scaler = MinMaxScaler()
user_course_matrix_scaled = scaler.fit_transform(user_course_matrix)

svd = TruncatedSVD(n_components=50, random_state=42)
user_course_svd = svd.fit_transform(user_course_matrix_scaled)

def recommend_courses_collab(user_id, top_n=5):
    if user_id not in train_data["user_id"].values:
        return "User tidak memiliki cukup data untuk rekomendasi."

    user_idx = train_data[train_data["user_id"] == user_id].index[0]
    user_vector = user_course_svd[user_idx].reshape(1, -1)

    similarities = cosine_similarity(user_vector, user_course_svd).flatten()
    course_indices = np.argsort(similarities)[::-1][:top_n]

    return courses_df.iloc[course_indices][["course_id", "title"]].to_dict(orient="records")

# ============================================= EVALUASI MODEL (RMSE) =============================================
test_user_course_matrix = test_data.pivot(index="user_id", columns="course_id_unq", values="ratings").fillna(0)
test_user_course_matrix_scaled = scaler.transform(test_user_course_matrix)

predicted_ratings = svd.transform(test_user_course_matrix_scaled)
rmse = np.sqrt(mean_squared_error(test_user_course_matrix_scaled, predicted_ratings))
print(f"RMSE: {rmse}")

# ============================================= FLASK API =============================================
@app.route('/recommend/content', methods=['GET'])
def recommend_content():
    course_id = request.args.get('course_id')
    if not course_id:
        return jsonify({"error": "Parameter course_id diperlukan"}), 400
    
    recommendations = recommend_courses_content(course_id)
    return jsonify(recommendations)

@app.route('/recommend/collab', methods=['GET'])
def recommend_collab():
    user_id = request.args.get('user_id', type=int)
    if not user_id:
        return jsonify({"error": "Parameter user_id diperlukan"}), 400

    recommendations = recommend_courses_collab(user_id)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
