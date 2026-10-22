"""
database.py — SbobinAI
Gestisce l'inizializzazione del database SQLite tramite SQLAlchemy.
Tabelle: subjects, lectures, bookmarks
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer, String, Text,
    Float, create_engine, text, event
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

# ── Percorso DB Dinamico ────────────────────────────────────────────────────────
from config import ensure_storage_dirs, get_db_path

dirs = ensure_storage_dirs()
DB_PATH = dirs["db_path"]
DATABASE_URL = f"sqlite:///{DB_PATH}"

@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA busy_timeout=5000;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.close()

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def rebind_database() -> None:
    """Riconfigura l'engine e la SessionLocal se il percorso di archiviazione cambia."""
    global engine, SessionLocal, DB_PATH, DATABASE_URL
    dirs = ensure_storage_dirs()
    DB_PATH = dirs["db_path"]
    DATABASE_URL = f"sqlite:///{DB_PATH}"
    try:
        engine.dispose()
    except Exception:
        pass
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    SessionLocal.configure(bind=engine)
    init_db()
    print(f"[DB] Database ri-collegato → {DB_PATH}")



# ── Base ORM ───────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── Modelli ────────────────────────────────────────────────────────────────────

class Subject(Base):
    """Materia universitaria (es. Analisi II, Storia dell'Arte)."""
    __tablename__ = "subjects"

    id: int = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name: str = Column(String(255), nullable=False, unique=True)
    description: Optional[str] = Column(Text, nullable=True)
    # Parole chiave che guidano Whisper nel riconoscimento del vocabolario tecnico
    whisper_prompt_keywords: Optional[str] = Column(Text, nullable=True)
    color: str = Column(String(7), default="#6366f1")  # HEX color per la UI
    created_at: datetime = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    lectures = relationship("Lecture", back_populates="subject", cascade="all, delete-orphan")


class Lecture(Base):
    """Singola lezione registrata."""
    __tablename__ = "lectures"

    id: int = Column(Integer, primary_key=True, index=True, autoincrement=True)
    subject_id: int = Column(Integer, ForeignKey("subjects.id", ondelete="CASCADE"), nullable=False, index=True)
    title: str = Column(String(255), nullable=False, default="Lezione senza titolo")
    date: datetime = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    # Trascrizione grezza da Whisper
    raw_transcript: Optional[str] = Column(Text, nullable=True)
    # Note elaborate dall'LLM (Quaderno Digitale)
    processed_notes: Optional[str] = Column(Text, nullable=True)
    audio_path: Optional[str] = Column(String(512), nullable=True)
    duration_seconds: Optional[float] = Column(Float, nullable=True)
    source_type: str = Column(String(50), default="recorded")  # 'recorded' | 'imported'
    in_quaderno: bool = Column(Boolean, default=True)
    # Domande rilevate del docente (JSON serializzato)
    detected_questions: Optional[str] = Column(Text, nullable=True)
    created_at: datetime = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    subject = relationship("Subject", back_populates="lectures")
    bookmarks = relationship("Bookmark", back_populates="lecture", cascade="all, delete-orphan")


class Bookmark(Base):
    """Segnalibro esame — momento preciso di una lezione."""
    __tablename__ = "bookmarks"

    id: int = Column(Integer, primary_key=True, index=True, autoincrement=True)
    lecture_id: int = Column(Integer, ForeignKey("lectures.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp_seconds: float = Column(Float, nullable=False)
    note: Optional[str] = Column(Text, nullable=True)
    created_at: datetime = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    lecture = relationship("Lecture", back_populates="bookmarks")


# ── Init DB ────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Crea tutte le tabelle se non esistono."""
    Base.metadata.create_all(bind=engine)
    # Migrazioni leggere per DB creati da versioni precedenti (es. unicopilot.db)
    migrations = [
        ("detected_questions", "TEXT"),
        ("source_type", "VARCHAR(50) DEFAULT 'recorded'"),
        ("in_quaderno", "BOOLEAN DEFAULT 1"),
    ]
    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(lectures)"))}
        for col, ddl in migrations:
            if col not in existing:
                try:
                    conn.execute(text(f"ALTER TABLE lectures ADD COLUMN {col} {ddl}"))
                    conn.commit()
                except Exception:
                    pass
    print(f"[DB] Database inizializzato → {DB_PATH}")


init_db()


def get_db():
    """Dependency injection di FastAPI per la sessione DB."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
