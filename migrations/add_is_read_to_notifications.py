from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def upgrade():
    db = SessionLocal()
    try:
        db.execute(text("ALTER TABLE notifications ADD COLUMN is_read BOOLEAN DEFAULT 0"))
        db.commit()
        print("Successfully added is_read column to notifications table")
    except Exception as e:
        db.rollback()
        print(f"Error during migration: {str(e)}")
        raise e
    finally:
        db.close()

def downgrade():
    db = SessionLocal()
    try:
        db.execute(text("ALTER TABLE notifications DROP COLUMN is_read"))
        db.commit()
        print("Successfully removed is_read column from notifications table")
    except Exception as e:
        db.rollback()
        print(f"Error during migration: {str(e)}")
        raise e
    finally:
        db.close()

if __name__ == "__main__":
    upgrade() 