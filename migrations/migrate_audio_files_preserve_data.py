import sqlite3
import os

DB_PATH = os.getenv("DATABASE_URL", "app.db").replace("sqlite:///", "")

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# 1. Create new table with correct schema
c.execute('''
CREATE TABLE audio_files_new (
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

# 2. Copy data from old table to new table
# We'll use COALESCE to provide default values for missing columns
c.execute('''
INSERT INTO audio_files_new (id, title, filename, transcription, transcription_with_timestamps, date, user_id, created_at)
SELECT 
    id,
    '' AS title, -- default empty string for missing title
    filename,
    transcription,
    transcription_with_timestamps,
    CURRENT_TIMESTAMP AS date, -- default to now for missing date
    user_id,
    created_at
FROM audio_files
''')

# 3. Drop old table
c.execute('DROP TABLE audio_files')

# 4. Rename new table to original name
c.execute('ALTER TABLE audio_files_new RENAME TO audio_files')

# 5. Recreate index
c.execute('CREATE INDEX IF NOT EXISTS ix_audio_files_id ON audio_files (id)')

conn.commit()
conn.close()

print("Migration complete. Data preserved and schema updated.") 