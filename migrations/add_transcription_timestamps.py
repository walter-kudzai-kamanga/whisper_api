from sqlalchemy import create_engine, Column, Text, text
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
        # Add the new column
        db.execute(text("ALTER TABLE audio_files ADD COLUMN transcription_with_timestamps TEXT"))
        db.commit()
        print("Successfully added transcription_with_timestamps column")
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
        # Remove the column
        db.execute(text("ALTER TABLE audio_files DROP COLUMN transcription_with_timestamps"))
        db.commit()
        print("Successfully removed transcription_with_timestamps column")
    except Exception as e:
        db.rollback()
        print(f"Error during migration: {str(e)}")
        raise e
    finally:
        db.close()

if __name__ == "__main__":
    upgrade() 