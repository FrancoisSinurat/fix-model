from routes.auth import router as auth_router
from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import TruncatedSVD
from auth import get_current_user
from sync_to_db import save_courses_to_db, save_reviews_to_db, save_combine_df_to_db, sync_users_from_reviews
from databases import User
from google.cloud import storage
import re
from io import StringIO
import math
import os
import logging

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)

# Konfigurasi Google Cloud Storage

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp-key.json"
storage_client = storage.Client()
bucket_name = "sinity-bucket"
# Cache dataset
course_df = None
users = None
scaled_ratings_df = None
cosine_sim_matrix = None

logging.basicConfig(level=logging.INFO)



def read_csv_from_gcs(file_name):
    """Mengambil file CSV dari Google Cloud Storage."""
    try:
        logging.info(f"Mengambil {file_name} dari Google Cloud Storage...")
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        data = blob.download_as_text()
        logging.info(f"Berhasil mengambil {file_name}!")
        return pd.read_csv(StringIO(data))
    except Exception as e:
        logging.error(f"Error loading {file_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading {file_name}: {str(e)}")


def load_data():
    """Memuat dataset dari GCS dan melakukan preprocessing."""
    global course_df, users, scaled_ratings_df, cosine_sim_matrix, combine_df  

    try:
        course_df = read_csv_from_s3("Coursera_courses.csv")
        print("Berhasil baca Coursera_courses.csv")
        reviews_df = read_csv_from_s3("Coursera_reviews.csv")
        print("Berhasil baca Coursera_reviews.csv")
    except HTTPException as e:
        logging.error(e.detail)
        return

    # Preprocessing courses
    if 'Unnamed: 0' in course_df.columns:
        course_df.drop(columns=['Unnamed: 0'], inplace=True)
    course_df.drop(columns=['institution', 'course_url'], inplace=True, errors='ignore')
    
    
    def normalize_text(text):
        text = re.sub(r'[^a-zA-Z:=,\s]', '', text)
        text = text.lower()
        words = text.split()
        return ' '.join(words)
    
    course_df = course_df.dropna(subset=["name"])
    course_df["name"] = course_df["name"].apply(normalize_text)
    
    label_encoder = LabelEncoder()
    course_df["course_id_int"] = label_encoder.fit_transform(course_df["course_id"])
    
    

    # Preprocessing Reviews
    reviews_df.drop(columns=['reviews','date_reviews'], inplace=True)
    valid_users = reviews_df['reviewers'].value_counts(dropna=True)[lambda x: x >= 2].index
    users = reviews_df[reviews_df['reviewers'].isin(valid_users)].copy()
    users = users.sample(n=min(20000, len(users)), random_state=42)
    user_id_mapping = {user: idx for idx, user in enumerate(users['reviewers'].unique())}
    users['user_id'] = users['reviewers'].map(user_id_mapping)
    users['course_id_int'] = users['course_id'].map(course_df.set_index('course_id')['course_id_int'])
    users.dropna(subset=['course_id_int'], inplace=True)
    users = users.drop_duplicates(subset=["user_id", "course_id_int"], keep="last")
    
    valid_course_ids = users['course_id'].unique()
    course_df = course_df[course_df['course_id'].isin(valid_course_ids)]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(2, 2))
    tfidf_matrix = vectorizer.fit_transform(course_df["name"])
    cosine_sim_matrix = cosine_similarity(tfidf_matrix)
    
    users_count_filtered = reviews_df.groupby("course_id")["reviewers"].nunique().reset_index()
    users_count_filtered.rename(columns={"reviewers": "total_reviewers"}, inplace=True)

    average_rating_filtered = reviews_df.groupby("course_id")["rating"].mean().reset_index()
    average_rating_filtered.rename(columns={"rating": "average_rating"}, inplace=True)

    combine_df = course_df.copy()
    combine_df = combine_df.merge(users_count_filtered, on="course_id", how="left")
    combine_df = combine_df.merge(average_rating_filtered, on="course_id", how="left")

    combine_df["total_reviewers"] = combine_df["total_reviewers"].fillna(0).astype(int)
    combine_df["average_rating"] = combine_df["average_rating"].fillna(0).round(2)

    # Simpan data ke DB
    save_courses_to_db(course_df)
    save_reviews_to_db(users)
    save_combine_df_to_db(combine_df)
    
    # Collaborative Filtering Preparation
    rating_matrix = users.pivot_table(index='user_id', columns='course_id_int', values='rating').fillna(0)
    scaler = MinMaxScaler()
    scaled_matrix = scaler.fit_transform(rating_matrix)
    scaled_ratings_df = pd.DataFrame(scaled_matrix, index=rating_matrix.index, columns=rating_matrix.columns)

    # Dimensionality Reduction
    svd = TruncatedSVD(n_components=50, random_state=42)
    reduced_matrix = svd.fit_transform(scaled_ratings_df)
    scaled_ratings_df = pd.DataFrame(reduced_matrix, index=rating_matrix.index)

    logging.info("SVD dan scaling rating selesai.")
    logging.info("Dataset berhasil dimuat dan diproses!")
    sync_users_from_reviews()


# Load dataset pertama kali
load_data()

@app.get("/")
def read_root():
    return {"message": "Halo, ini adalah API rekomendasi kursus dibuat oleh Francois Novalentino S"}



@app.get("/courses", response_model=dict)
async def get_all_courses():
    if combine_df is None:
        raise HTTPException(status_code=500, detail="Dataset belum dimuat.")
    courses = combine_df[["name", "course_id", "course_id_int", "total_reviewers", "average_rating"]].to_dict(orient="records")

    return {"courses": courses}


@app.get("/users", response_model=dict)
def get_users(current_user: User = Depends(get_current_user)):
    user_id = current_user.user_id
    if users is None or combine_df is None:
        raise HTTPException(status_code=500, detail="Dataset belum dimuat.")

    # Ambil data pengguna berdasarkan user_id
    user_reviews = users[users["user_id"] == user_id]

    if user_reviews.empty:
        raise HTTPException(status_code=404, detail="User tidak ditemukan atau belum cukup banyak memberi rating.")

    # Ambil nama reviewer
    reviewer_name = user_reviews["reviewers"].iloc[0]

    # Ambil daftar kursus yang dirating oleh user tersebut
    rated_courses = user_reviews[["course_id", "course_id_int", "rating"]]

    # Gabungkan dengan `combine_df` untuk mendapatkan informasi kursus
    user_courses = rated_courses.merge(
        combine_df[["course_id", "name", "total_reviewers", "average_rating"]],
        on="course_id",
        how="left"
    )

    # Format data yang akan dikembalikan
    return {
        "user_id": user_id,
        "reviewer_name": reviewer_name,
        "rated_courses": [
            {
                "course_id": row["course_id"],
                "name": row["name"],
                "rating": row["rating"],
                "total_reviewers": row["total_reviewers"],
                "average_rating": row["average_rating"]
            }
            for _, row in user_courses.iterrows()
        ]
    }


@app.get("/recommend_course", response_model=dict)
async def recommend_courses(course_name: str = Query(..., title="Nama Kursus")):
    if course_df is None or cosine_sim_matrix is None:
        raise HTTPException(status_code=500, detail="Dataset belum dimuat.")

    course_name = course_name.lower().strip()
    matched_courses = course_df[course_df["name"].str.contains(course_name, case=False, na=False)]

    if matched_courses.empty:
        return {"message": f"Kursus '{course_name}' tidak ditemukan.", "recommendations": []}

    idx = matched_courses.index[0]  # Index asli dari course_df
    idx_pos = course_df.index.get_loc(idx)  # Posisi numerik dari index di DataFrame

    sim_scores = list(enumerate(cosine_sim_matrix[idx_pos]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    recommendations = []
    for i in sim_scores:
        if i[0] < len(course_df):  # Hindari out-of-bounds
            course = course_df.iloc[i[0]]
            recommendations.append({
                "course_id": course["course_id"],
                "name": course["name"],
                "similarity": round(i[1], 4)
            })

    return {
        "search_name": course_name,
        "recommendations": recommendations
    }


@app.get("/recommend_for_user", response_model=dict)
async def recommend_for_user(current_user: User = Depends(get_current_user)):
    user_id = current_user.user_id
    if scaled_ratings_df is None or users is None or combine_df is None:
        raise HTTPException(status_code=500, detail="Dataset belum dimuat.")

    # Cek apakah user_id ada di rating matrix
    if user_id not in scaled_ratings_df.index:
        # Jika tidak ada, anggap user baru → tampilkan top courses
        top_rated_courses = combine_df.sort_values(by="total_reviewers", ascending=False).head(5)
        return {
            "message": "User baru, menampilkan kursus terpopuler.",
            "recommendations": [
                {
                    "course_id": row["course_id"],
                    "name": row["name"],
                    "total_reviews": row["total_reviewers"],
                    "average_rating": row["average_rating"],
                }
                for _, row in top_rated_courses.iterrows()
            ]
        }

    # Ambil data user
    user = users[users['user_id'] == user_id]
    reviewer_name = user['reviewers'].iloc[0] if not user.empty else "Unknown"

    # Ambil course yang sudah pernah dirating oleh user
    rated_courses = set(user['course_id_int'])

    # Ambil top predicted scores dari matrix, exclude yang sudah dirating
    top_courses = scaled_ratings_df.loc[user_id].drop(labels=rated_courses, errors='ignore').sort_values(ascending=False).head(5)

    # Ambil detail course dari rekomendasi
    recommended_courses = combine_df[combine_df['course_id_int'].isin(top_courses.index)][['course_id', 'name', 'total_reviewers', 'average_rating']]

    return {
        "user_id": user_id,
        "reviewer": reviewer_name,
        "recommendations": [
            {
                "course_id": row["course_id"],
                "name": row["name"],
                "total_reviews": row["total_reviewers"],
                "average_rating": row["average_rating"],
            }
            for _, row in recommended_courses.iterrows()
        ]
    }


# ==========================
#   Hybrid Recommendation
# ==========================
@app.get("/hybrid_recommendation", response_model=dict)
async def hybrid_recommendation(alpha: float = Query(0.5, ge=0, le=1), current_user: User = Depends(get_current_user)):
    user_id = current_user.user_id

    if scaled_ratings_df is None or cosine_sim_matrix is None:
        raise HTTPException(status_code=500, detail="Dataset belum dimuat.")
    
    if user_id not in scaled_ratings_df.index:
        raise HTTPException(status_code=404, detail="User ID tidak ditemukan")

    user_ratings = scaled_ratings_df.loc[user_id]
    rated_courses = set(users[users['user_id'] == user_id]['course_id_int'])

    if not rated_courses:
        raise HTTPException(status_code=400, detail="User belum memiliki riwayat rating.")

    # Collaborative
    collab_scores = user_ratings.copy()
    collab_scores = collab_scores.reindex(combine_df['course_id_int'], fill_value=0)
    collab_scores_array = collab_scores.values.reshape(-1, 1)
    collab_scores_scaled = MinMaxScaler((0, 5)).fit_transform(collab_scores_array).flatten()
    collab_series = pd.Series(collab_scores_scaled, index=combine_df.index)


    # Content-based
    cb_scores = np.zeros(len(combine_df))
    for course_int in rated_courses:
        cb_scores += cosine_sim_matrix[course_int]
    cb_scores /= len(rated_courses)

    cb_series = pd.Series(cb_scores, index=combine_df.index)
    scaler = MinMaxScaler((0, 5))
    cb_series_scaled = scaler.fit_transform(cb_series.values.reshape(-1, 1)).flatten()
    cb_series = pd.Series(cb_series_scaled, index=cb_series.index)

    # Hybrid
    hybrid_scores = alpha * collab_series + (1 - alpha) * cb_series
    hybrid_series = pd.Series(hybrid_scores, index=combine_df.index)

    rated_indexes = combine_df[combine_df['course_id_int'].isin(rated_courses)].index
    hybrid_series = hybrid_series.drop(rated_indexes, errors='ignore')

    top_indexes = hybrid_series.sort_values(ascending=False).head(5).index
    recommended = combine_df.loc[top_indexes][['course_id', 'name', 'total_reviewers', 'average_rating']]


    return {
        "user_id": user_id,
        "alpha": alpha,
        "recommendations": [
            {
                "course_id": row["course_id"],
                "name": row["name"],
                "total_reviewers": row["total_reviewers"],
                "average_rating": row["average_rating"]
            }
            for _, row in recommended.iterrows()
        ]
    }

# ==========================
#     Evaluate Model
# ==========================
def evaluate_model():
    """
    Menghitung MAE dan RMSE antara rating asli dan prediksi dari scaled_ratings_df
    """
    if users is None or scaled_ratings_df is None:
        raise Exception("Dataset belum dimuat.")

    y_true = []
    y_pred = []
    skipped = 0  # Counter untuk data yang dilewati
    
    for _, row in users.iterrows():
        user_id = row['user_id']
        course_id_int = int(row['course_id_int'])

        if user_id in scaled_ratings_df.index and course_id_int in scaled_ratings_df.columns:
            y_true.append(row['rating'])
            y_pred.append(scaled_ratings_df.loc[user_id, course_id_int])
        else:
            skipped += 1
            
    if len(y_true) == 0:
        raise Exception("Tidak ada sampel yang bisa dievaluasi.")
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "Total_samples": len(y_true)
    }

@app.get("/evaluate_model")
def get_model_evaluation():
    """
    Endpoint untuk mendapatkan hasil evaluasi model (MAE & RMSE)
    """
    try:
        results = evaluate_model()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def precision_at_k(y_true, y_pred, k=5):
    """
    Menghitung Precision@k
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Menyortir prediksi untuk mendapatkan top-k rekomendasi
    top_k_predictions = np.argsort(y_pred)[-k:]
    top_k_true = np.array([1 if i in top_k_predictions else 0 for i in range(len(y_pred))])

    # Menghitung Precision@k
    precision = np.sum(top_k_true * y_true) / k
    return precision

def recall_at_k(y_true, y_pred, k=5):
    """
    Menghitung Recall@k
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Menyortir prediksi untuk mendapatkan top-k rekomendasi
    top_k_predictions = np.argsort(y_pred)[-k:]
    top_k_true = np.array([1 if i in top_k_predictions else 0 for i in range(len(y_pred))])

    # Menghitung Recall@k
    recall = np.sum(top_k_true * y_true) / np.sum(y_true)
    return recall

@app.get("/evaluate_recommendation")
def evaluate_recommendation(k: int = Query(5, ge=1, le=10)):
    """
    Endpoint untuk mendapatkan evaluasi model dengan precision@k dan recall@k
    """
    if users is None or scaled_ratings_df is None:
        raise HTTPException(status_code=500, detail="Dataset belum dimuat.")
    
    y_true = []
    y_pred = []
    
    for _, row in users.iterrows():
        user_id = row['user_id']
        course_id_int = row['course_id_int']
        
        if user_id in scaled_ratings_df.index and course_id_int in scaled_ratings_df.columns:
            # Menganggap rating >= 4 adalah relevan
            y_true.append(1 if row['rating'] >= 4 else 0)
            y_pred.append(scaled_ratings_df.loc[user_id, course_id_int])
    
    precision = precision_at_k(y_true, y_pred, k)
    recall = recall_at_k(y_true, y_pred, k)

    return {
        "Precision@k": round(precision, 4),
        "Recall@k": round(recall, 4),
        "k": k,
        "Total_samples": len(y_true)
    }


# ==========================
#   Alias untuk Frontend
# ==========================
@app.get("/recommendations", response_model=dict)
async def get_recommendations(alpha: float = Query(0.5, ge=0, le=1), current_user: User = Depends(get_current_user)):
    """
    Alias endpoint dari hybrid_recommendation untuk konsumsi frontend
    """
    
    try:
        return await hybrid_recommendation(user_id=current_user.id, alpha=alpha)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
