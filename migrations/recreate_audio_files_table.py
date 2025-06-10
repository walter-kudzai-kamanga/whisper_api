import sqlite3
import os

DB_PATH = os.getenv("DATABASE_URL", "app.db").replace("sqlite:///", "")

def run_migration():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        # Create backup of existing data
        c.execute("CREATE TABLE audio_files_backup AS SELECT * FROM audio_files")
        
        # Drop the existing table and its index
        c.execute("DROP TABLE IF EXISTS audio_files")
        c.execute("DROP INDEX IF EXISTS ix_audio_files_id")
        
        # Create new table with exact schema matching SQLAlchemy model
        c.execute('''
        CREATE TABLE audio_files (
            id INTEGER PRIMARY KEY NOT NULL,
            title VARCHAR NOT NULL,
            filename VARCHAR NOT NULL,
            transcription TEXT NOT NULL,
            transcription_with_timestamps TEXT,
            date DATETIME NOT NULL,
            user_id INTEGER,
            created_at DATETIME,
            FOREIGN KEY(user_id) REFERENCES users (id)
        )
        ''')
        
        # Create index
        c.execute('CREATE INDEX ix_audio_files_id ON audio_files (id)')
        
        # Restore data from backup
        c.execute('''
        INSERT INTO audio_files (id, title, filename, transcription, date, user_id, created_at)
        SELECT id, title, filename, transcription, date, user_id, created_at
        FROM audio_files_backup
        ''')
        
        # Drop backup table
        c.execute("DROP TABLE audio_files_backup")
        
        conn.commit()
        print("Successfully recreated audio_files table with correct schema")
    except Exception as e:
        conn.rollback()
        print(f"Error during migration: {str(e)}")
        raise e
    finally:
        conn.close()

if __name__ == "__main__":
    run_migration() 