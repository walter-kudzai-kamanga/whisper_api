from database import SessionLocal, init_db, Base, engine
from models import Admin
from auth import get_password_hash
from datetime import datetime
from sqlalchemy import inspect

def create_super_user():
    # First ensure database is initialized
    print("Initializing database...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    # Verify tables were created
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"Created tables: {tables}")
    
    db = SessionLocal()
    try:
        # Create super admin user
        password = "admin1234"
        hashed_password = get_password_hash(password)
        print(f"Creating super user with email: admin@walter.com")
        print(f"Username: Walter")
        print(f"Password hash: {hashed_password}")
        
        # Check if admin already exists
        existing_admin = db.query(Admin).filter(Admin.email == "admin@walter.com").first()
        if existing_admin:
            print("Super user already exists, updating...")
            existing_admin.hashed_password = hashed_password
            existing_admin.username = "Walter"
            existing_admin.is_active = True
            existing_admin.is_super_admin = True
            db.commit()
            print("Super user updated successfully!")
        else:
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
            print("Super user created successfully!")
        
        # Verify the admin was created/updated
        admin = db.query(Admin).filter(Admin.email == "admin@walter.com").first()
        if admin:
            print(f"Super user verified in database:")
            print(f"Email: {admin.email}")
            print(f"Username: {admin.username}")
            print(f"Stored password hash: {admin.hashed_password}")
            print(f"is_active: {admin.is_active}")
            print(f"is_super_admin: {admin.is_super_admin}")
            print(f"created_at: {admin.created_at}")
            print(f"last_login: {admin.last_login}")
            
            # Test login verification
            from auth import verify_password
            if verify_password(password, admin.hashed_password):
                print("Password verification test passed!")
            else:
                print("WARNING: Password verification test failed!")
        else:
            print("Warning: Super user not found in database after creation!")
            
    except Exception as e:
        print(f"Error creating super user: {str(e)}")
        db.rollback()
        raise e
    finally:
        db.close()

if __name__ == "__main__":
    # Create super user
    create_super_user() 