import sqlite3
import os

DB_PATH = os.getenv("DATABASE_URL", "app.db").replace("sqlite:///", "")

def run_migration():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        # First, check if the column exists
        c.execute("PRAGMA table_info(audio_files)")
        columns = [column[1] for column in c.fetchall()]
        
        if 'transcription_with_timestamps' not in columns:
            # Add the new column
            c.execute("ALTER TABLE audio_files ADD COLUMN transcription_with_timestamps TEXT")
            conn.commit()
            print("Successfully added transcription_with_timestamps column")
        else:
            print("Column already exists")
            
    except Exception as e:
        conn.rollback()
        print(f"Error during migration: {str(e)}")
        raise e
    finally:
        conn.close()

if __name__ == "__main__":
    run_migration() 