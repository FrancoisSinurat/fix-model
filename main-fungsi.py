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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

def clean_text(text):
    try:
        if detect(text) != 'en':
            return ""
    except:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove symbols
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text
def preprocess_courses(courses_df):
    """Preprocess dataset courses: hapus kolom tidak perlu & encode course_id."""
    courses_df = courses_df.drop(columns=['institution', 'course_url'], errors='ignore')
    
    # Encoding course_id
    label_encoder = LabelEncoder()
    courses_df['course_id_unq'] = label_encoder.fit_transform(courses_df['course_id'])

    # Mapping course_id ke course_id_unq
    course_id_mapping = pd.DataFrame({
        'course_id': courses_df['course_id'],
        'course_id_unq': courses_df['course_id_unq']
    })

    # Simpan hasil preprocessing ke CSV
    courses_df.to_csv("fix_courses.csv", index=False)
    print(f"Preprocessing Courses berhasil: \n {courses_df}")

    return courses_df, course_id_mapping

def preprocess_reviews(reviews_df, course_id_mapping):
    """Preprocess dataset reviews: hapus kolom tidak perlu, sesuaikan course_id, dan buat user_id unik."""
    reviews_df = reviews_df.drop(columns=['date_reviews'], errors='ignore')
    # Sesuaikan course_id di reviews agar sesuai dengan course_id_unq
    reviews_df = reviews_df.merge(course_id_mapping, on='course_id', how='inner')

    # Menentukan user yang memberikan review minimal 2 kursus
    user_counts = reviews_df['reviewers'].value_counts()
    multiple_ratings = user_counts[user_counts >= 2].index

    # Filter dataset berdasarkan reviewer yang memenuhi syarat
    reviews_filtered = reviews_df[reviews_df['reviewers'].isin(multiple_ratings)]
    
    # Menghapus duplikasi berdasarkan pasangan (reviewers, course_id)
    reviews_filtered1 = reviews_filtered.drop_duplicates(subset=['reviewers', 'course_id'])
    reviews_filtered1 = reviews_filtered1.drop(columns=['course_id'], errors='ignore')
    # Membuat mapping user_id unik berdasarkan nama reviewer
    user_id_mapping = {}
    current_id = 1  # Mulai user_id dari 1

    for reviewer in reviews_df['reviewers']:
        if reviewer not in user_id_mapping:
            user_id_mapping[reviewer] = current_id
            current_id += 1

    # Menyalin data untuk menghindari SettingWithCopyWarning
    reviews_filtered1 = reviews_filtered1.copy()
    reviews_filtered1['user_id'] = reviews_filtered1['reviewers'].map(user_id_mapping)

    # Simpan hasil preprocessing ke CSV

    reviews_filtered1.to_csv("fix_reviews.csv", index=False)
    print(f"Preprocessing Reviews berhasil: \n {reviews_filtered1}")

    return reviews_filtered1

# Load dataset
courses_df = pd.read_csv("dataset-fix-SinityCourse/Coursera_courses.csv")
reviews_df = pd.read_csv("dataset-fix-SinityCourse/Coursera_reviews.csv")

# Jalankan preprocessing
courses_df, course_id_mapping = preprocess_courses(courses_df)
reviews_filtered1 = preprocess_reviews(reviews_df, course_id_mapping)
