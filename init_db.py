from database import SessionLocal, init_db
from models import Admin
from auth import get_password_hash
from datetime import datetime
from sqlalchemy import inspect

def init_admin():
    db = SessionLocal()
    try:
        # Verify admin table exists
        inspector = inspect(db.get_bind())
        if "admins" not in inspector.get_table_names():
            raise Exception("Admin table not found in database!")
        
        # Create super admin user
        password = "admin1234"
        hashed_password = get_password_hash(password)
        print(f"Creating admin with email: admin@walter.com")
        print(f"Password hash: {hashed_password}")
        
        # Check if admin already exists
        existing_admin = db.query(Admin).filter(Admin.email == "admin@walter.com").first()
        if existing_admin:
            print("Admin already exists, updating password...")
            existing_admin.hashed_password = hashed_password
            existing_admin.username = "admin1234"
            existing_admin.is_active = True
            existing_admin.is_super_admin = True
            db.commit()
            print("Admin password updated successfully!")
        else:
            admin = Admin(
                email="admin@walter.com",
                username="admin1234",
                hashed_password=hashed_password,
                created_at=datetime.utcnow(),
                is_active=True,
                is_super_admin=True
            )
            db.add(admin)
            db.commit()
            print("Super admin user created successfully!")
        
        # Verify the admin was created/updated
        admin = db.query(Admin).filter(Admin.email == "admin@walter.com").first()
        if admin:
            print(f"Admin verified in database: {admin.email}")
            print(f"Stored password hash: {admin.hashed_password}")
            print(f"Admin username: {admin.username}")
            print(f"Admin is_active: {admin.is_active}")
            print(f"Admin is_super_admin: {admin.is_super_admin}")
            
            # Test login verification
            from auth import verify_password
            if verify_password(password, admin.hashed_password):
                print("Password verification test passed!")
            else:
                print("WARNING: Password verification test failed!")
        else:
            print("Warning: Admin not found in database after creation!")
            
    except Exception as e:
        print(f"Error creating admin user: {str(e)}")
        db.rollback()
        raise e
    finally:
        db.close()

if __name__ == "__main__":
    # Initialize database
    init_db()
    # Create admin user
    init_admin() 