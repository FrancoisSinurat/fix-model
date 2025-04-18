from sqlalchemy.orm import Session
from databases import session, Course, UserReview, CombineDf
import logging

def save_courses_to_db(course_df):
    try:
        for _, row in course_df.iterrows():
            course = Course(
                course_id_int=row['course_id_int'],
                name=row['name'],
                course_id=row['course_id']
            )
            session.add(course)
        session.commit()
        logging.info("Data courses berhasil disimpan ke database!")
    except Exception as e:
        session.rollback()
        logging.error(f"Gagal menyimpan data courses: {e}")

def save_reviews_to_db(users_df):
    try:
        for _, row in users_df.iterrows():
            review = UserReview(
                user_id=row['user_id'],
                rating=row['rating'],
                review=row['review'],
                course_id_int=row['course_id_int']
            )
            session.add(review)
        session.commit()
        logging.info("Data reviews berhasil disimpan ke database!")
    except Exception as e:
        session.rollback()
        logging.error(f" Gagal menyimpan data reviews: {e}")

def save_combine_df_to_db(combine_df):
    try:
        for _, row in combine_df.iterrows():
            combine_data = CombineDf(
                course_id_int=row['course_id_int'],
                total_reviews=row['total_reviews'],
                average_rating=row['average_rating']
            )
            session.add(combine_data)
        session.commit()
        logging.info("Data combine_df berhasil disimpan ke database!")
    except Exception as e:
        session.rollback()
        logging.error(f"Gagal menyimpan data combine_df: {e}")
