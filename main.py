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

# Load Data
courses_df = pd.read_csv("dataset-fix-SinityCourse/Coursera_courses.csv")  # Course
reviews_df = pd.read_csv("dataset-fix-SinityCourse/Coursera_reviews.csv")  # Reviews

def clean_text(text):
    try:
        if not text or detect(text) != 'en':  # Hapus jika tidak ada review atau bukan bahasa Inggris
            return ""
    except:
        return ""
    
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Membagi teks berdasarkan akhir kalimat
    selected_sentences = " ".join(sentences[:3])  # Ambil maksimal 3 kalimat pertama

    # Hapus simbol, kecuali titik, tanda tanya, dan seru untuk tetap mempertahankan kalimat
    selected_sentences = re.sub(r"[^a-zA-Z0-9\s.!?]", "", selected_sentences)
    selected_sentences = re.sub(r"\s+", " ", selected_sentences).strip()  # Menghapus spasi berlebih
    
    return selected_sentences

reviews_df['reviews'] = reviews_df['reviews'].astype(str).apply(clean_text) 
# ================================================ PREPROCESSING COURSE ====================================================

# Menghapus kolom yang tidak diperlukan
courses_df.drop(columns=['institution', 'course_url'], inplace=True)

# Encoding course_id
label_encoder = LabelEncoder()
courses_df['course_id_unq'] = label_encoder.fit_transform(courses_df['course_id'])

# Membuat mapping course_id ke course_id_unq
course_id_mapping = pd.DataFrame({
    'course_id': courses_df['course_id'],
    'course_id_unq': courses_df['course_id_unq']
})

# Menghapus kolom asli course_id setelah encoding
# courses_df.drop(columns=['course_id'], inplace=True)

# Simpan hasil preprocessing courses ke CSV
courses_df.to_csv("fix_courses.csv", index=False)


# ================================================ PREPROCESSING REVIEWER ====================================================
# Menghapus kolom yang tidak diperlukan
reviews_df.drop(columns=['date_reviews'], inplace=True)

# Menyesuaikan course_id di reviews agar sesuai dengan course_id_unq di courses_df
reviews_df = reviews_df.merge(course_id_mapping, on='course_id', how='left')
# reviews_df.rename(columns={'course_id_unq': 'course_id_unq'}, inplace=True)

# Menentukan user yang telah memberikan review untuk setidaknya 2 kursus berbeda
user_counts = reviews_df['reviewers'].value_counts()
multiple_ratings = user_counts[user_counts >= 2].index

# Membuat mapping user_id baru untuk user yang memenuhi syarat
user_id_mapping = {reviewer: idx + 1 for idx, reviewer in enumerate(multiple_ratings)}

# Filter dataset ulasan berdasarkan reviewer yang memenuhi syarat
reviews_filtered = reviews_df[reviews_df['reviewers'].isin(multiple_ratings)]

# Menghapus duplikasi data berdasarkan pasangan (reviewers, course_id)
reviews_filtered1 = reviews_filtered.drop_duplicates(subset=['reviewers', 'course_id'])

# Membuat mapping user_id unik berdasarkan nama reviewer
user_id_mapping = {}
current_id = 1  # Mulai user_id dari 1

# Iterasi melalui reviewer untuk memastikan user_id tetap konsisten jika nama sama
for reviewer in reviews_df['reviewers']:
    if reviewer not in user_id_mapping:
        user_id_mapping[reviewer] = current_id
        current_id += 1

# Menyalin data untuk menghindari SettingWithCopyWarning
reviews_filtered1 = reviews_filtered1.copy()
reviews_filtered1['user_id'] = reviews_filtered1['reviewers'].map(user_id_mapping)

# Simpan hasil preprocessing reviews ke CSV


# Apply text cleaning to reviews
reviews_filtered1.to_csv("fix_reviews.csv", index=False)
print("Reviews berhasil dibersihkan")

# ================================================ REVIEWS SAMPLE ====================================================

# reviews_sampled = reviews_filtered1.sample(n=100000, random_state=42)
