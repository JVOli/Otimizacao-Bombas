import os
from sqlalchemy import create_engine, Column, Integer, String, Float, ARRAY
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.environ.get("DATABASE_URL", "")

Base = declarative_base()


class CustomPump(Base):
    __tablename__ = "custom_pumps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False, unique=True)
    vazao = Column(ARRAY(Float), nullable=False)
    altura = Column(ARRAY(Float), nullable=False)
    rendimento = Column(ARRAY(Float), nullable=False, default=[])


engine = None
SessionLocal = None


def init_db():
    global engine, SessionLocal
    if not DATABASE_URL:
        return
    url = DATABASE_URL
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    engine = create_engine(url, pool_pre_ping=True)
    SessionLocal = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)


def get_db():
    if SessionLocal is None:
        yield None
        return
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
