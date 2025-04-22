import logging
from databases import session, Course, UserReview, CombineDf, User
from auth import get_password_hash

def save_courses_to_db(course_df):
    try:
        courses = [
            Course(
                course_id_int=row['course_id_int'],
                name=row['name'],
                course_id=row['course_id']
            )
            for _, row in course_df.iterrows()
        ]
        session.bulk_save_objects(courses)
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error save_courses: {e}")

def save_reviews_to_db(users_df):
    try:
        reviews = [
            UserReview(
                user_id=row['user_id'],
                rating=row['rating'],
                reviewers=row['reviewers'],
                course_id_int=row['course_id_int']
            )
            for _, row in users_df.iterrows()
        ]
        session.bulk_save_objects(reviews)
        session.commit()
        logging.info(f"Successfully saved {len(users_df)} user reviews to DB.")
    except Exception as e:
        session.rollback()
        logging.error(f"Error save_reviews: {e}")

def save_combine_df_to_db(combine_df):
    try:
        combine_data = [
            CombineDf(
                course_id_int=row['course_id_int'],
                total_reviewers=row['total_reviewers'],
                average_rating=row['average_rating']
            )
            for _, row in combine_df.iterrows()
        ]
        session.bulk_save_objects(combine_data)
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error save_combine_df: {e}")

def sync_users_from_reviews(df_users):
    try:
        new_users = [
            User(
                user_id=row["user_id"],
                name=(row["reviewers"].strip() if row["reviewers"] else "Unknown Reviewer"),
                email=f"user{row['user_id']}@example.com",
                password=get_password_hash("password123")
            )
            for _, row in df_users.iterrows()
        ]
        session.bulk_save_objects(new_users)
        session.commit()
        logging.info(f"[SYNC] {len(new_users)} users inserted to User table.")
    except Exception as e:
        session.rollback()
        logging.error(f"[SYNC ERROR] sync_users_from_reviews: {e}")
