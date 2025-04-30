"""Database configuration and connection management."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL from environment variable, default to SQLite for development
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///classifier.db')

# Create engine with appropriate configuration based on database type
if DATABASE_URL.startswith('sqlite:'):
    # SQLite doesn't support pooling
    engine = create_engine(
        DATABASE_URL,
        echo=False,  # Set to True to see SQL queries
    )
else:
    # For other databases (PostgreSQL, MySQL, etc.), use connection pooling
    engine = create_engine(
        DATABASE_URL,
        echo=False,  # Set to True to see SQL queries
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
    )

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create thread-safe session factory
db_session = scoped_session(SessionLocal)

# Base class for declarative models
Base = declarative_base()

def get_db():
    """Get database session."""
    db = db_session()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database, creating all tables."""
    Base.metadata.create_all(bind=engine)