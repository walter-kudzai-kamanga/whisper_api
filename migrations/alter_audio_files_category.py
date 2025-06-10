import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime, ForeignKey, text
from database import SQLALCHEMY_DATABASE_URL
from datetime import datetime

def upgrade():
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    metadata = MetaData()
    metadata.reflect(bind=engine)

    # 1. Create a new table with the correct schema
    with engine.connect() as conn:
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS audio_files_new (
                id INTEGER PRIMARY KEY,
                title VARCHAR NOT NULL,
                filename VARCHAR NOT NULL,
                transcription VARCHAR NOT NULL,
                date DATETIME NOT NULL,
                category VARCHAR NULL,
                user_id INTEGER,
                created_at DATETIME
            )
        '''))
        # 2. Copy data from old table to new table, providing default for NULL transcriptions
        conn.execute(text('''
            INSERT INTO audio_files_new (id, title, filename, transcription, date, category, user_id, created_at)
            SELECT id, title, filename, COALESCE(transcription, '') as transcription, date, category, user_id, created_at FROM audio_files
        '''))
        # 3. Drop the old table
        conn.execute(text('DROP TABLE audio_files'))
        # 4. Rename the new table
        conn.execute(text('ALTER TABLE audio_files_new RENAME TO audio_files'))
        conn.commit()

if __name__ == "__main__":
    upgrade() 