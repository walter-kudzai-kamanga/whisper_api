import os
import subprocess
from database import Base, engine
from models import Admin
from auth import get_password_hash
from datetime import datetime
from sqlalchemy import inspect

def setup_database():
    print("Setting up database...")
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)
    
    # Verify tables
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"Available tables: {tables}")
    
    # Create super admin only if it doesn't exist
    from database import SessionLocal
    db = SessionLocal()
    try:
        # Check if super admin exists
        existing_admin = db.query(Admin).filter(Admin.email == "admin@walter.com").first()
        if not existing_admin:
            # Create super admin user
            password = "admin1234"
            hashed_password = get_password_hash(password)
            print(f"Creating super admin with email: admin@walter.com")
            print(f"Username: Walter")
            print(f"Password hash: {hashed_password}")
            
            admin = Admin(
                email="admin@walter.com",
                username="Walter",
                hashed_password=hashed_password,
                created_at=datetime.utcnow(),
                is_active=True,
                is_super_admin=True,
                last_login=datetime.utcnow()
            )
            db.add(admin)
            db.commit()
            print("Super admin created successfully!")
        else:
            print("Super admin already exists, skipping creation.")
            
    except Exception as e:
        print(f"Error in database setup: {str(e)}")
        db.rollback()
        raise e
    finally:
        db.close()

def start_server():
    print("Starting server...")
    # Kill any existing uvicorn processes
    subprocess.run(["pkill", "-f", "uvicorn"])
    # Start the server
    subprocess.run(["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])

if __name__ == "__main__":
    # Create instance directory if it doesn't exist
    os.makedirs("instance", exist_ok=True)
    
    # Setup database and create super admin if needed
    setup_database()
    
    # Start server
    start_server() 