from sqlalchemy import create_engine, Column, String, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Get the database URL from environment or use default
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")

# Create engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def upgrade():
    # Create a session
    db = SessionLocal()
    
    try:
        # Add the new columns
        db.execute(text("ALTER TABLE audio_files ADD COLUMN author VARCHAR"))
        db.execute(text("ALTER TABLE audio_files ADD COLUMN category VARCHAR"))
        db.commit()
        print("Successfully added author and category columns")
    except Exception as e:
        db.rollback()
        print(f"Error during migration: {str(e)}")
        raise e
    finally:
        db.close()

def downgrade():
    # Create a session
    db = SessionLocal()
    
    try:
        # Remove the columns
        db.execute(text("ALTER TABLE audio_files DROP COLUMN author"))
        db.execute(text("ALTER TABLE audio_files DROP COLUMN category"))
        db.commit()
        print("Successfully removed author and category columns")
    except Exception as e:
        db.rollback()
        print(f"Error during migration: {str(e)}")
        raise e
    finally:
        db.close()

if __name__ == "__main__":
    upgrade() 