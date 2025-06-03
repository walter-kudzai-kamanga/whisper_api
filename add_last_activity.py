from sqlalchemy import create_engine, Column, DateTime
from sqlalchemy.ext.declarative import declarative_base
from database import engine
import sqlite3

def add_last_activity_column():
    # Connect to the SQLite database
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    
    try:
        # Add the last_activity column
        cursor.execute('ALTER TABLE users ADD COLUMN last_activity DATETIME')
        conn.commit()
        print("Successfully added last_activity column")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("Column already exists")
        else:
            print(f"Error: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":
    add_last_activity_column() 