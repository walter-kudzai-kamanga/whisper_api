from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models
from auth import get_password_hash

def create_first_admin():
    db = SessionLocal()
    try:
        # Check if any admin exists
        admin = db.query(models.Admin).first()
        if admin:
            print("Admin account already exists!")
            return

        # Create first admin
        admin = models.Admin(
            email="admin@walter.com",
            username="superadmin",
            hashed_password=get_password_hash("admin1234"),
            is_active=True
        )
        db.add(admin)
        db.commit()
        print("First admin account created successfully!")
        print("Email: admin@walter.com")
        print("Username: superadmin")
        print("Password: admin1234")
    except Exception as e:
        print(f"Error creating admin: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    # Create tables
    models.Base.metadata.create_all(bind=engine)
    # Create first admin
    create_first_admin() 