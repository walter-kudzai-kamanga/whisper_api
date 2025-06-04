from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, BackgroundTasks, Form
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import whisper
import shutil
import os
from uuid import uuid4
from datetime import timedelta, datetime
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content
import jinja2
import os
from dotenv import load_dotenv
import ssl
import mimetypes
import json
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile
import numpy as np
import signal
from contextlib import contextmanager
import time

from database import engine, get_db
import models
import schemas
from auth import (
    verify_password,
    get_password_hash,
    create_access_token,
    get_current_active_user,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    SECRET_KEY,
    ALGORITHM,
    create_refresh_token
)
from jose import JWTError, jwt

# Load environment variables
load_dotenv()

app = FastAPI(title="Audio Transcription API")

# Configure OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Whisper model
device = "cuda" if torch.cuda.is_available() else "cpu"
# Use tiny model for faster processing
model = whisper.load_model("tiny", device=device)

# Audio processing constants
SAMPLE_RATE = 16000  # Whisper expects 16kHz audio

# Email configuration
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
FROM_EMAIL = os.getenv("FROM_EMAIL", "your-verified-sender@yourdomain.com")

# Email template loader
template_loader = jinja2.FileSystemLoader(searchpath="./templates")
template_env = jinja2.Environment(loader=template_loader)

# Define supported audio formats
SUPPORTED_AUDIO_FORMATS = {
    'audio/mpeg': '.mp3',
    'audio/mp4': '.m4a',
    'audio/wav': '.wav',
    'audio/x-wav': '.wav',
    'audio/ogg': '.ogg',
    'audio/webm': '.webm',
    'audio/aac': '.aac',
    'audio/flac': '.flac'
}

# Create a thread pool for CPU-bound tasks
thread_pool = ThreadPoolExecutor(max_workers=4)

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

async def process_audio_file(
    file_path: str,
    chunk_size: int = 30,  # Process 30 seconds at a time
    overlap: int = 5  # 5 seconds overlap between chunks
):
    try:
        # Load audio file
        audio = whisper.load_audio(file_path)
        duration = len(audio) / SAMPLE_RATE
        print(f"Processing audio file of duration: {duration:.2f} seconds")
        
        # Split into chunks with overlap
        chunks = []
        chunk_starts = []
        for i in range(0, len(audio), int(chunk_size * SAMPLE_RATE)):
            chunk_end = min(i + int((chunk_size + overlap) * SAMPLE_RATE), len(audio))
            chunk = audio[i:chunk_end]
            chunks.append(chunk)
            chunk_starts.append(i / SAMPLE_RATE)
        
        print(f"Split audio into {len(chunks)} chunks")
        
        # Process chunks sequentially with time limit for each
        results = []
        for idx, chunk in enumerate(chunks):
            print(f"Processing chunk {idx + 1}/{len(chunks)}")
            start_time = time.time()
            
            try:
                with time_limit(55):  # 55 seconds timeout per chunk
                    result = model.transcribe(
                        chunk,
                        language="en",
                        fp16=(device == "cuda"),
                        verbose=False,
                        beam_size=1,  # Reduce beam size for speed
                        best_of=1,    # Reduce best_of for speed
                        temperature=0 # Disable temperature for speed
                    )
                    results.append({
                        "text": result["text"],
                        "start_time": chunk_starts[idx],
                        "duration": len(chunk) / SAMPLE_RATE
                    })
            except TimeoutError:
                print(f"Chunk {idx + 1} timed out, skipping...")
                continue
            
            processing_time = time.time() - start_time
            print(f"Chunk {idx + 1} completed in {processing_time:.2f} seconds")
        
        if not results:
            raise Exception("No chunks were successfully processed")
        
        # Combine results, handling overlaps
        final_text = []
        for i, result in enumerate(results):
            if i > 0:
                # Add a space between chunks
                final_text.append(" ")
            final_text.append(result["text"])
        
        return {
            "text": "".join(final_text),
            "duration": duration,
            "language": "en",  # Assuming English for all chunks
            "num_chunks": len(chunks),
            "successful_chunks": len(results)
        }
    except Exception as e:
        print(f"Error in process_audio_file: {str(e)}")
        raise e

def get_file_extension(content_type: str) -> str:
    """Get file extension from content type"""
    return SUPPORTED_AUDIO_FORMATS.get(content_type, '.mp3')

def send_email(to_email: str, subject: str, template_name: str, **kwargs):
    if not SENDGRID_API_KEY:
        print(f"Email not sent to {to_email}: SendGrid API key not configured")
        return
        
    template = template_env.get_template(template_name)
    html_content = template.render(**kwargs)
    
    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        message = Mail(
            from_email=Email(FROM_EMAIL),
            to_emails=To(to_email),
            subject=subject,
            html_content=Content("text/html", html_content)
        )
        response = sg.send(message)
        print(f"Email sent successfully to {to_email}")
        print(f"Status code: {response.status_code}")
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        # Continue with registration even if email fails
        pass

@app.post("/register/", response_model=schemas.User)
async def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already taken")
    
    hashed_password = get_password_hash(user.password)
    db_user = models.User(
        email=user.email,
        username=user.username,
        hashed_password=hashed_password,
        roles=user.role,
        created_at=datetime.utcnow(),
        is_active=True,
        is_verified=False
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Create notification for new account
    notification = models.Notification(
        type="new_account",
        content=f"New user registered: {user.username}",
        extra_data=json.dumps({"user_id": db_user.id, "email": user.email})
    )
    db.add(notification)
    db.commit()
    
    # Send welcome email (will be skipped if SendGrid is not configured)
    send_email(
        user.email,
        "Welcome to Audio Transcription API",
        "welcome.html",
        username=user.username
    )
    
    return db_user

@app.post("/token", response_model=schemas.Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/password-reset/")
async def request_password_reset(
    reset_request: schemas.PasswordReset,
    db: Session = Depends(get_db)
):
    user = db.query(models.User).filter(models.User.email == reset_request.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    reset_token = create_access_token(data={"sub": user.username}, expires_delta=timedelta(hours=1))
    
    send_email(
        user.email,
        "Password Reset Request",
        "password_reset.html",
        username=user.username,
        reset_token=reset_token
    )
    
    return {"message": "Password reset email sent"}

@app.post("/password-update/")
async def update_password(
    update_data: schemas.PasswordUpdate,
    db: Session = Depends(get_db)
):
    try:
        payload = jwt.decode(update_data.token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=400, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=400, detail="Invalid token")
    
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.hashed_password = get_password_hash(update_data.new_password)
    db.commit()
    
    return {"message": "Password updated successfully"}

@app.post("/audio-files/", response_model=schemas.AudioFile)
async def create_audio_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(...),
    category: str = Form(...),
    author: str = Form(...),
    duration: str = Form(...),
    date: datetime = Form(...),
    image: str = Form(...),
    text_document: str = Form(None),
    db: Session = Depends(get_db)
):
    try:
        if not file:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided"
            )
        
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No filename provided"
            )
        
        # Validate file type
        if not file.content_type or file.content_type not in SUPPORTED_AUDIO_FORMATS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type. Supported types are: {', '.join(SUPPORTED_AUDIO_FORMATS.keys())}"
            )
        
        try:
            # Create a temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Generate temp filename with appropriate extension
                extension = get_file_extension(file.content_type)
                temp_filename = os.path.join(temp_dir, f"temp_{uuid4().hex}{extension}")
                
                # Save the file
                with open(temp_filename, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                # Process the audio file
                result = await process_audio_file(temp_filename)
                
                # Create audio file record
                audio_file = models.AudioFile(
                    title=title,
                    filename=os.path.basename(temp_filename),
                    transcription=result["text"],
                    category=category,
                    author=author,
                    duration=duration,
                    date=date,
                    image=image,
                    text_document=text_document,
                    user_id=None
                )
                db.add(audio_file)
                db.commit()
                db.refresh(audio_file)

                return audio_file
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing file: {str(e)}"
            )
    except Exception as e:
        print(f"Unexpected error in create_audio_file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )

@app.get("/audio-files/", response_model=List[schemas.AudioFile])
async def get_all_audio_files(
    skip: int = 0,
    limit: int = 100,
    category: Optional[str] = None,
    author: Optional[str] = None,
    db: Session = Depends(get_db)
):
    try:
        print("Fetching audio files...")
        query = db.query(models.AudioFile)
        
        # Apply filters if provided
        if category:
            query = query.filter(models.AudioFile.category == category)
        if author:
            query = query.filter(models.AudioFile.author == author)
        
        # Order by date descending (newest first)
        audio_files = query.order_by(models.AudioFile.date.desc()).offset(skip).limit(limit).all()
        print(f"Found {len(audio_files)} audio files")
        
        # Debug print first file if exists
        if audio_files:
            print(f"First file data: {audio_files[0].__dict__}")
        
        return audio_files
    except Exception as e:
        print(f"Error in get_all_audio_files: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching audio files: {str(e)}"
        )

@app.get("/users/me/", response_model=schemas.User)
async def read_users_me(current_user: models.User = Depends(get_current_active_user)):
    return current_user

@app.post("/login/", response_model=schemas.Token)
async def login(
    login_data: schemas.LoginRequest,
    db: Session = Depends(get_db)
):
    print(f"Login attempt with email: {login_data.email}")
    
    # First try admin login
    admin = db.query(models.Admin).filter(models.Admin.email == login_data.email).first()
    if admin:
        print(f"Found admin: {admin.email}")
        if verify_password(login_data.password, admin.hashed_password):
            print("Admin password verified!")
            # Update last login
            admin.last_login = datetime.utcnow()
            db.commit()
            
            # Create both access and refresh tokens
            access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = create_access_token(
                data={"sub": admin.username}, expires_delta=access_token_expires
            )
            refresh_token = create_refresh_token(data={"sub": admin.username})
            
            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer"
            }
        else:
            print("Admin password verification failed!")
    else:
        print("No admin found with that email")
    
    # If not admin, try user login
    user = db.query(models.User).filter(models.User.email == login_data.email).first()
    if user:
        print(f"Found user: {user.email}")
        if verify_password(login_data.password, user.hashed_password):
            print("User password verified!")
            # Update last activity
            user.last_activity = datetime.utcnow()
            db.commit()
            
            # Create both access and refresh tokens
            access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = create_access_token(
                data={"sub": user.username}, expires_delta=access_token_expires
            )
            refresh_token = create_refresh_token(data={"sub": user.username})
            
            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer"
            }
        else:
            print("User password verification failed!")
    else:
        print("No user found with that email")
    
    # If neither admin nor user found, raise unauthorized
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect email or password",
        headers={"WWW-Authenticate": "Bearer"},
    )

@app.post("/refresh-token/", response_model=schemas.Token)
async def refresh_token(
    refresh_token: str,
    db: Session = Depends(get_db)
):
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = db.query(models.User).filter(models.User.username == username).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create new access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

# Admin authentication
async def get_current_admin(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = schemas.TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    admin = db.query(models.Admin).filter(models.Admin.username == token_data.username).first()
    if admin is None:
        raise credentials_exception
    if not admin.is_active:
        raise HTTPException(status_code=400, detail="Inactive admin")
    return admin

@app.post("/admin/register/", response_model=schemas.Admin)
async def register_admin(
    admin: schemas.AdminCreate,
    db: Session = Depends(get_db)
):
    # Check if email or username already exists
    db_admin = db.query(models.Admin).filter(models.Admin.email == admin.email).first()
    if db_admin:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    db_admin = db.query(models.Admin).filter(models.Admin.username == admin.username).first()
    if db_admin:
        raise HTTPException(status_code=400, detail="Username already taken")
    
    # Create new admin
    hashed_password = get_password_hash(admin.password)
    db_admin = models.Admin(
        email=admin.email,
        username=admin.username,
        hashed_password=hashed_password,
        created_at=datetime.utcnow()
    )
    db.add(db_admin)
    db.commit()
    db.refresh(db_admin)
    
    return db_admin

@app.post("/admin/login/", response_model=schemas.Token)
async def admin_login(
    login_data: schemas.LoginRequest,
    db: Session = Depends(get_db)
):
    print(f"Admin login attempt with email: {login_data.email}")
    print(f"Password received: {login_data.password}")
    
    # Find admin
    admin = db.query(models.Admin).filter(models.Admin.email == login_data.email).first()
    if admin:
        print(f"Found admin: {admin.email}")
        print(f"Stored hash: {admin.hashed_password}")
        print(f"Username: {admin.username}")
        print(f"is_active: {admin.is_active}")
        print(f"is_super_admin: {admin.is_super_admin}")
        
        # Verify password
        if verify_password(login_data.password, admin.hashed_password):
            print("Admin password verified!")
            # Update last login
            admin.last_login = datetime.utcnow()
            db.commit()
            
            # Create both access and refresh tokens
            access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = create_access_token(
                data={"sub": admin.username}, expires_delta=access_token_expires
            )
            refresh_token = create_refresh_token(data={"sub": admin.username})
            
            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer"
            }
        else:
            print("Admin password verification failed!")
    else:
        print("No admin found with that email")
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect email or password",
        headers={"WWW-Authenticate": "Bearer"},
    )

@app.get("/admin/me/", response_model=schemas.Admin)
async def read_admin_me(current_admin: models.Admin = Depends(get_current_admin)):
    return current_admin

@app.get("/admin/list/", response_model=List[schemas.Admin])
async def list_admins(
    current_admin: models.Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    return db.query(models.Admin).all()

@app.delete("/admin/{admin_id}/")
async def delete_admin(
    admin_id: int,
    current_admin: models.Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    if current_admin.id == admin_id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    
    admin = db.query(models.Admin).filter(models.Admin.id == admin_id).first()
    if not admin:
        raise HTTPException(status_code=404, detail="Admin not found")
    
    db.delete(admin)
    db.commit()
    return {"message": "Admin deleted successfully"}

@app.post("/notifications/", response_model=schemas.Notification)
async def create_notification(
    notification: schemas.NotificationCreate,
    db: Session = Depends(get_db)
):
    db_notification = models.Notification(
        type=notification.type,
        content=notification.content,
        extra_data=notification.extra_data
    )
    db.add(db_notification)
    db.commit()
    db.refresh(db_notification)
    return db_notification

@app.get("/notifications/", response_model=List[schemas.Notification])
async def get_notifications(
    skip: int = 0,
    limit: int = 100,
    notification_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(models.Notification)
    if notification_type:
        query = query.filter(models.Notification.type == notification_type)
    notifications = query.order_by(models.Notification.created_at.desc()).offset(skip).limit(limit).all()
    return notifications

@app.get("/public/users/", response_model=List[schemas.PublicUser])
async def list_all_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    users = db.query(models.User).filter(models.User.created_at.isnot(None)).offset(skip).limit(limit).all()
    return users

@app.get("/public/admins/", response_model=List[schemas.PublicAdmin])
async def list_all_admins(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    admins = db.query(models.Admin).filter(models.Admin.created_at.isnot(None)).offset(skip).limit(limit).all()
    return admins

@app.get("/public/all-users/")
async def list_all_users_and_admins(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    users = db.query(models.User).filter(models.User.created_at.isnot(None)).offset(skip).limit(limit).all()
    admins = db.query(models.Admin).filter(models.Admin.created_at.isnot(None)).offset(skip).limit(limit).all()
    users_out = [schemas.PublicUser.model_validate(u) for u in users]
    admins_out = [schemas.PublicAdmin.model_validate(a) for a in admins]
    return {
        "users": users_out,
        "admins": admins_out
    }

@app.post("/transcribe/")
async def transcribe_audio(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided"
        )
    
    # Validate file type
    if not file.content_type or file.content_type not in SUPPORTED_AUDIO_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Supported types are: {', '.join(SUPPORTED_AUDIO_FORMATS.keys())}"
        )
    
    try:
        # Create a temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate temp filename with appropriate extension
            extension = get_file_extension(file.content_type)
            temp_filename = os.path.join(temp_dir, f"temp_{uuid4().hex}{extension}")
            
            # Save the file
            with open(temp_filename, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Process the audio file
            result = await process_audio_file(temp_filename)
            
            return {
                "filename": file.filename,
                "transcription": result["text"],
                "duration": result["duration"],
                "language": result["language"]
            }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )
