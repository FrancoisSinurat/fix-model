import os
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# Ambil konfigurasi dari environment variables
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")  # default ke 5432
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@/{DB_NAME}?host=/cloudsql/{DB_HOST}"


# Base dan engine setup
Base = declarative_base()
engine = create_engine(DATABASE_URL)
Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
session = Session()

# Model Tabel
class Course(Base):
    __tablename__ = 'courses'

    course_id_int = Column(Integer, primary_key=True)
    name = Column(String)
    course_id = Column(String, unique=True)

    reviews = relationship("UserReview", back_populates="course")
    combine_data = relationship("CombineDf", back_populates="course")


class UserReview(Base):
    __tablename__ = 'user_reviews'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    rating = Column(Float)
    review = Column(String)
    course_id_int = Column(Integer, ForeignKey("courses.course_id_int"))

    user = relationship("User", back_populates="reviews")
    course = relationship("Course", back_populates="reviews")



class CombineDf(Base):
    __tablename__ = 'combine_df'

    rating_id = Column(Integer, primary_key=True)
    course_id_int = Column(Integer, ForeignKey('courses.course_id_int'))
    total_reviews = Column(Integer)
    average_rating = Column(Float)

    course = relationship("Course", back_populates="combine_data")



# Hanya saat kamu ingin reset total
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)
