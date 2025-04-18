from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import TruncatedSVD
from google.cloud import storage
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

# Konfigurasi Google Cloud Storage
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp-key.json"
storage_client = storage.Client()
bucket_name = "sinitycourse-dataset"

# Cache dataset
course_df = None
reviews_filtered = None
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
    global course_df, reviews_filtered, scaled_ratings_df, cosine_sim_matrix

    try:
        course_df = read_csv_from_gcs("Coursera_courses.csv")
        reviews_df = read_csv_from_gcs("Coursera_reviews.csv")
    except HTTPException as e:
        logging.error(e.detail)
        return

    # Preprocessing courses
    course_df.drop(columns=['institution', 'course_url'], inplace=True, errors='ignore')
    course_df = course_df.dropna(subset=["name"])
    course_df["name"] = course_df["name"].astype(str).str.replace(r'["\']', '', regex=True).str.strip().str.lower()

    # Encoding `course_id`
    label_encoder = LabelEncoder()
    course_df["course_id_int"] = label_encoder.fit_transform(course_df["course_id"])

    # TF-IDF untuk Content-Based Filtering
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(2, 2))
    tfidf_matrix = vectorizer.fit_transform(course_df["name"])
    cosine_sim_matrix = cosine_similarity(tfidf_matrix) # 

    # Collaborative Filtering
    # Menghitung jumlah unik reviewers per course di reviews_filtered
    reviewer_counts_filtered = reviews_df.groupby("course_id")["reviewers"].nunique().reset_index()
    reviewer_counts_filtered.rename(columns={"reviewers": "total_reviewers"}, inplace=True)

    # Menghitung rata-rata rating per course
    average_rating_filtered = reviews_df.groupby("course_id")["rating"].mean().reset_index()
    average_rating_filtered.rename(columns={"rating": "average_rating"}, inplace=True)

    # Gabungkan total reviewers dan rata-rata rating ke dalam course_df
    course_df = course_df.merge(reviewer_counts_filtered, on="course_id", how="left")
    course_df = course_df.merge(average_rating_filtered, on="course_id", how="left")
    
    valid_users = reviews_df['reviewers'].value_counts()[lambda x: x >= 2].index
    

    reviews_filtered = reviews_df[reviews_df['reviewers'].isin(valid_users)].copy()
    
    # membatasi dataset ke 10.000 baris karena terlalu besar
    reviews_filtered = reviews_filtered.sample(n=min(30000, len(reviews_filtered)), random_state=42)

    # mapping user_id dari reviewers
    user_id_mapping = {user: idx for idx, user in enumerate(reviews_filtered['reviewers'].unique())}
    reviews_filtered['user_id'] = reviews_filtered['reviewers'].map(user_id_mapping)
    
    # menghapus kolom yang tidak digunakan 'date_reviews'
    reviews_filtered.drop(columns=['date_reviews'], inplace=True)

    # mapping course_id ke course_id_int
    reviews_filtered['course_id_int'] = reviews_filtered['course_id'].map(course_df.set_index('course_id')['course_id_int'])
    reviews_filtered.dropna(subset=['course_id_int'], inplace=True)


    # Isi nilai NaN dengan 0 untuk kursus yang belum memiliki review
    course_df["total_reviewers"] = course_df["total_reviewers"].fillna(0).astype(int)
    course_df["average_rating"] = course_df["average_rating"].fillna(0).round(2)  # Dibulatkan ke 2 desimal
    
    pivot_df = reviews_filtered.pivot_table(index='user_id', columns='course_id_int', values='rating', fill_value=0)

   # Matrix Factorization dengan Truncated SVD
    n_components = 20
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_features = svd.fit_transform(pivot_df)
    item_features = svd.components_

    predicted_ratings = np.dot(user_features, item_features)

    scaler = MinMaxScaler(feature_range=(0, 5))
    scaled_ratings = scaler.fit_transform(predicted_ratings)
    global scaled_ratings_df
    scaled_ratings_df = pd.DataFrame(scaled_ratings, index=pivot_df.index, columns=pivot_df.columns)

    logging.info("Dataset berhasil dimuat dan diproses!")


# Load dataset pertama kali
load_data()


@app.get("/")
def read_root():
    return {"message": "Halo, ini adalah API rekomendasi kursus dibuat oleh Francois Novalentino S"}


@app.get("/refresh_data")
async def refresh_data():
    """Memperbarui dataset dari GCS."""
    load_data()
    return {"message": "Dataset berhasil diperbarui"}

@app.get("/courses", response_model=dict)
async def get_all_courses():
    if course_df is None:
        raise HTTPException(status_code=500, detail="Dataset belum dimuat.")
    courses = course_df[["name", "course_id","total_reviewers","average_rating"]].to_dict(orient="records")
    return {"courses": courses}


@app.get("/users", response_model=dict)
def get_users(user_id: int = Query(..., title="User ID")):
    if reviews_filtered is None or course_df is None:
        raise HTTPException(status_code=500, detail="Dataset belum dimuat.")

    # Ambil data pengguna berdasarkan user_id
    user_reviews = reviews_filtered[reviews_filtered["user_id"] == user_id]

    if user_reviews.empty:
        raise HTTPException(status_code=404, detail="User tidak ditemukan atau belum cukup banyak memberi rating.")

    # Ambil nama reviewer
    reviewer_name = user_reviews["reviewers"].iloc[0]

    # Ambil daftar kursus yang dirating oleh user tersebut
    rated_courses = user_reviews[["course_id", "course_id_int", "rating"]]

    # Gabungkan dengan `course_df` untuk mendapatkan informasi kursus
    user_courses = rated_courses.merge(
        course_df[["course_id", "name", "total_reviewers", "average_rating"]],
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

    idx = matched_courses.index[0]
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    recommendations = [
        {
            "course_id": course_df.iloc[i[0]]["course_id"],
            "name": course_df.iloc[i[0]]["name"],
            "similarity": round(i[1], 4)
        } for i in sim_scores
    ]

    return {
        "search_name": course_name,
        "recommendations": recommendations
    }


@app.get("/recommend_for_user", response_model=dict)
async def recommend_for_user(user_id: int = Query(..., title="User ID")):
    if scaled_ratings_df is None or reviews_filtered is None:
        raise HTTPException(status_code=500, detail="Dataset belum dimuat.")

    if user_id not in scaled_ratings_df.index:
        top_rated_courses = course_df.sort_values(by="total_reviewers", ascending=False).head(5)
        return {
            "message": "User baru, menampilkan kursus secara acak.",
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

    user = reviews_filtered[reviews_filtered['user_id'] == user_id]
    reviewer_name = user['reviewers'].iloc[0] if not user.empty else None  # Ambil nama reviewer

    rated_courses = set(reviews_filtered[reviews_filtered['user_id'] == user_id]['course_id_int'])
    top_courses = scaled_ratings_df.loc[user_id].sort_values(ascending=False)
    top_courses = top_courses[~top_courses.index.isin(rated_courses)].head(5)

    recommended_courses = course_df[course_df['course_id_int'].isin(top_courses.index)][['course_id', 'name', 'total_reviewers', 'average_rating']]

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

# Hybrid filtering
# ==========================
#     Evaluate Model
# ==========================
def evaluate_model():
    """
    Menghitung MAE dan RMSE antara rating asli dan prediksi dari scaled_ratings_df
    """
    if reviews_filtered is None or scaled_ratings_df is None:
        raise Exception("Dataset belum dimuat.")

    y_true = []
    y_pred = []

    for _, row in reviews_filtered.iterrows():
        user_id = row['user_id']
        course_id_int = int(row['course_id_int'])

        if user_id in scaled_ratings_df.index and course_id_int in scaled_ratings_df.columns:
            y_true.append(row['rating'])
            y_pred.append(scaled_ratings_df.loc[user_id, course_id_int])

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

# ==========================
#   Hybrid Recommendation
# ==========================
@app.get("/hybrid_recommendation", response_model=dict)
async def hybrid_recommendation(user_id: int = Query(...), alpha: float = Query(0.5, ge=0, le=1)):
    if scaled_ratings_df is None or cosine_sim_matrix is None:
        raise HTTPException(status_code=500, detail="Dataset belum dimuat.")
    
    if user_id not in scaled_ratings_df.index:
        raise HTTPException(status_code=404, detail="User ID tidak ditemukan")

    user_ratings = scaled_ratings_df.loc[user_id]
    rated_courses = set(reviews_filtered[reviews_filtered['user_id'] == user_id]['course_id_int'])

    if not rated_courses:
        raise HTTPException(status_code=400, detail="User belum memiliki riwayat rating.")

    # Collaborative
    collab_scores = user_ratings.copy()
    collab_scores = collab_scores.reindex(course_df['course_id_int'], fill_value=0)

    # Content-based
    cb_scores = np.zeros(len(course_df))
    for course_int in rated_courses:
        cb_scores += cosine_sim_matrix[course_int]
    cb_scores /= len(rated_courses)

    cb_series = pd.Series(cb_scores, index=course_df.index)
    scaler = MinMaxScaler((0, 5))
    cb_series_scaled = scaler.fit_transform(cb_series.values.reshape(-1, 1)).flatten()
    cb_series = pd.Series(cb_series_scaled, index=cb_series.index)

    # Hybrid
    hybrid_scores = alpha * cb_series + (1 - alpha) * collab_scores.values
    hybrid_series = pd.Series(hybrid_scores, index=course_df.index)

    rated_indexes = course_df[course_df['course_id_int'].isin(rated_courses)].index
    hybrid_series = hybrid_series.drop(rated_indexes, errors='ignore')

    top_indexes = hybrid_series.sort_values(ascending=False).head(5).index
    recommended = course_df.loc[top_indexes]

    return {
        "user_id": user_id,
        "alpha": alpha,
        "recommendations": [
            {
                "course_id": row["course_id"],
                "name": row["name"],
                "total_reviews": row["total_reviewers"],
                "average_rating": row["average_rating"]
            }
            for _, row in recommended.iterrows()
        ]
    }
# ==========================
#   Alias untuk Frontend
# ==========================
@app.get("/recommendations", response_model=dict)
async def get_recommendations(user_id: int, alpha: float = Query(0.5, ge=0, le=1)):
    """
    Alias endpoint dari hybrid_recommendation untuk konsumsi frontend
    """
    try:
        return await hybrid_recommendation(user_id=user_id, alpha=alpha)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
