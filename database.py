from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime

DATABASE_URL = "sqlite:///./eco.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Detection(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    file_hash = Column(String, nullable=False)
    car_detected = Column(Integer, default=0)
    plate_text = Column(String, nullable=True)
    confidence = Column(Float, default=0.0)
    duration_sec = Column(Float, default=0.0)

    # Новое поле для аудио
    audio_detected = Column(Integer, default=0)

    status = Column(String, default="pending")  # pending, confirmed, rejected
    ml_raw_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


# Пересоздаем таблицы (старые данные удалятся)
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)