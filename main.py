from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, BackgroundTasks, Form, Request
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
from fastapi.responses import FileResponse, JSONResponse
import requests

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

# SendGrid Email configuration
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
FROM_EMAIL = os.getenv("FROM_EMAIL", "your-verified-sender@yourdomain.com")
FROM_NAME = os.getenv("FROM_NAME", "Audio Transcription API")

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

AUDIO_STORAGE_PATH = os.path.join(os.path.dirname(__file__), "audio_storage")

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
        all_segments = []
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
                    
                    # Get segments with timestamps
                    segments = result.get("segments", [])
                    
                    # Adjust timestamps to account for chunk start time
                    for segment in segments:
                        segment["start"] += chunk_starts[idx]
                        segment["end"] += chunk_starts[idx]
                    
                    all_segments.extend(segments)
                    
                    chunk_time = time.time() - start_time
                    print(f"Chunk {idx + 1} processed in {chunk_time:.2f} seconds")
                    
            except TimeoutError:
                print(f"Chunk {idx + 1} timed out, skipping...")
                continue
            except Exception as e:
                print(f"Error processing chunk {idx + 1}: {str(e)}")
                continue
        
        # Combine all segments
        full_text = " ".join([segment.get("text", "").strip() for segment in all_segments])
        
        # Create transcription with timestamps
        transcription_with_timestamps = ""
        for segment in all_segments:
            try:
                start_time = int(segment.get("start", 0))
                end_time = int(segment.get("end", 0))
                text = segment.get("text", "").strip()
                if text:
                    transcription_with_timestamps += f"[{start_time:02d}:{end_time:02d}] {text}\n"
            except KeyError as e:
                print("KeyError:", e, "in segment:", segment)
        
        # Map segment keys for API response
        mapped_segments = []
        for segment in all_segments:
            mapped_segments.append({
                'start_time': segment.get('start', 0),
                'end_time': segment.get('end', 0),
                'text': segment.get('text', '')
            })
        
        return {
            "text": full_text,
            "segments": mapped_segments,
            "transcription_with_timestamps": transcription_with_timestamps.strip()
        }
        
    except Exception as e:
        print(f"Error in process_audio_file: {str(e)}")
        raise e

def get_file_extension(content_type: str) -> str:
    """Get file extension from content type"""
    return SUPPORTED_AUDIO_FORMATS.get(content_type, '.mp3')

async def send_email_async(to_email: str, subject: str, template_name: str, **kwargs) -> bool:
    """
    Send email using SendGrid asynchronously
    
    Args:
        to_email: Recipient email address
        subject: Email subject
        template_name: Name of the Jinja2 template file
        **kwargs: Template variables
        
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    if not SENDGRID_API_KEY:
        print(f"Email not sent to {to_email}: SendGrid API key not configured")
        return False
    
    try:
        # Render email template
        template = template_env.get_template(template_name)
        html_content = template.render(**kwargs)
        
        # Create SendGrid message
        message = Mail(
            from_email=Email(FROM_EMAIL, FROM_NAME),
            to_emails=To(to_email),
            subject=subject,
            html_content=Content("text/html", html_content)
        )
        
        # Send email
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        
        if response.status_code in [200, 201, 202]:
            print(f"Email sent successfully to {to_email} (Status: {response.status_code})")
            return True
        else:
            print(f"SendGrid API error: {response.status_code} - {response.body}")
            return False
            
    except Exception as e:
        print(f"Error sending email to {to_email}: {str(e)}")
        return False

def send_email_sync(to_email: str, subject: str, template_name: str, **kwargs) -> bool:
    """
    Send email using SendGrid synchronously (for backward compatibility)
    
    Args:
        to_email: Recipient email address
        subject: Email subject
        template_name: Name of the Jinja2 template file
        **kwargs: Template variables
        
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    if not SENDGRID_API_KEY:
        print(f"Email not sent to {to_email}: SendGrid API key not configured")
        return False
    
    try:
        # Render email template
        template = template_env.get_template(template_name)
        html_content = template.render(**kwargs)
        
        # Create SendGrid message
        message = Mail(
            from_email=Email(FROM_EMAIL, FROM_NAME),
            to_emails=To(to_email),
            subject=subject,
            html_content=Content("text/html", html_content)
        )
        
        # Send email
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        
        if response.status_code in [200, 201, 202]:
            print(f"Email sent successfully to {to_email} (Status: {response.status_code})")
            return True
        else:
            print(f"SendGrid API error: {response.status_code} - {response.body}")
            return False
            
    except Exception as e:
        print(f"Error sending email to {to_email}: {str(e)}")
        return False

@app.post("/register/", response_model=schemas.User)
async def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    try:
        print(f"Registration attempt for email: {user.email}, username: {user.username}")
        
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
        
        print(f"User registered successfully: {user.username} (ID: {db_user.id})")
        
        # Create notification for new account
        notification = models.Notification(
            type="new_account",
            content=f"New user registered: {user.username}",
            extra_data=json.dumps({"user_id": db_user.id, "email": user.email})
        )
        db.add(notification)
        db.commit()
        
        # Send welcome email asynchronously
        print(f"Sending welcome email to: {user.email}")
        email_success = await send_email_async(
            user.email,
            "Welcome to Audio Transcription API",
            "welcome.html",
            username=user.username
        )
        
        if email_success:
            print(f"Welcome email sent successfully to: {user.email}")
        else:
            print(f"Warning: Welcome email failed to send to: {user.email}")
            # Don't fail the registration if email fails
        
        return db_user
        
    except HTTPException:
        # Re-raise HTTP exceptions (like validation errors)
        raise
    except Exception as e:
        print(f"Unexpected error during registration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during registration. Please try again."
        )

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
    request: Request,
    db: Session = Depends(get_db)
):
    try:
        print(f"Password reset requested for email: {reset_request.email}")
        
        # Find user
        user = db.query(models.User).filter(models.User.email == reset_request.email).first()
        if not user:
            print(f"No user found with email: {reset_request.email}")
            # Don't reveal that the email doesn't exist
            return {"message": "If an account exists with this email, you will receive password reset instructions."}
        
        # Generate reset token
        reset_token = create_access_token(
            data={"sub": user.username},
            expires_delta=timedelta(hours=1)
        )
        
        # Create dynamic reset link based on the request
        base_url = str(request.base_url).rstrip('/')
        reset_link = f"{base_url}/reset-password/{reset_token}"
        
        print(f"Generated reset token for user: {user.username}")
        print(f"Reset link: {reset_link}")
        
        # Send password reset email using SendGrid
        success = await send_email_async(
            user.email,
            "Password Reset Request",
            "password_reset.html",
            username=user.username,
            reset_token=reset_token,
            reset_link=reset_link
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send password reset email. Please try again later."
            )
        
        print(f"Password reset email sent via SendGrid to: {user.email}")
        
        return {
            "message": "If an account exists with this email, you will receive password reset instructions.",
            "reset_link": reset_link
        }
    except Exception as e:
        print(f"Unexpected error in password reset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again later."
        )

@app.get("/reset-password/{token}")
async def reset_password_page(token: str):
    """
    Serve the password reset page with the token
    """
    try:
        # Verify the token is valid
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=400, detail="Invalid token")
        
        # Return the reset password HTML page
        return FileResponse("templates/reset_password_form.html")
        
    except JWTError:
        raise HTTPException(status_code=400, detail="Invalid or expired token")

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
    file: UploadFile = File(...),
    title: str = Form(...),
    transcription: str = Form(...),
    transcription_with_timestamps: Optional[str] = Form(None),  # Optional field
    date: datetime = Form(...),
    author: Optional[str] = Form(None),  # New field
    category: Optional[str] = Form(None),  # New field
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
            # Ensure audio_storage directory exists
            os.makedirs(AUDIO_STORAGE_PATH, exist_ok=True)
            # Generate a unique filename with appropriate extension
            extension = get_file_extension(file.content_type)
            unique_filename = f"{uuid4().hex}{extension}"
            file_path = os.path.join(AUDIO_STORAGE_PATH, unique_filename)
            # Save the file to audio_storage
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            # Create audio file record
            audio_file = models.AudioFile(
                title=title,
                filename=unique_filename,
                transcription=transcription,
                transcription_with_timestamps=transcription_with_timestamps,  # Can be None
                date=date,
                author=author,  # New field
                category=category,  # New field
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
    request: Request,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    try:
        print("Fetching audio files...")
        query = db.query(models.AudioFile)
        audio_files = query.order_by(models.AudioFile.date.desc()).offset(skip).limit(limit).all()
        print(f"Found {len(audio_files)} audio files")
        if not audio_files:
            return []
        result = []
        for audio_file in audio_files:
            item = audio_file.__dict__.copy()
            item["download_url"] = str(request.base_url) + f"audio-files/download/{audio_file.filename}"
            result.append(item)
        return result
    except Exception as e:
        print(f"Error in get_all_audio_files: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching audio files: {str(e)}"
        )

@app.get("/users/me/")
async def read_users_me():
    return {"message": "Authentication not required for this endpoint."}

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
    if not file.content_type in SUPPORTED_AUDIO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_AUDIO_FORMATS.keys())}"
        )
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=get_file_extension(file.content_type)) as temp_file:
        # Copy the uploaded file to the temporary file
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name
    
    try:
        # Process the audio file
        result = await process_audio_file(temp_file_path)
        
        # Create TranscriptionWithTimestamps object
        timestamped_transcription = schemas.TranscriptionWithTimestamps(
            segments=[
                schemas.TimestampSegment(
                    start_time=seg["start_time"],
                    end_time=seg["end_time"],
                    text=seg["text"]
                )
                for seg in result["segments"]
            ]
        )
        
        return {
            "transcription": result["text"],
            "transcription_with_timestamps": timestamped_transcription,
            "duration": result.get("duration", 0),
            "language": result.get("language", "en")
        }
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

@app.get("/audio-files/download/{filename}")
async def download_audio_file(filename: str):
    file_path = os.path.join(AUDIO_STORAGE_PATH, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="audio/mpeg", filename=filename)

@app.get("/audio-files/{sermon_id}")
async def get_audio_file_by_id(sermon_id: int, request: Request, db: Session = Depends(get_db)):
    audio_file = db.query(models.AudioFile).filter(models.AudioFile.id == sermon_id).first()
    if not audio_file:
        raise HTTPException(status_code=404, detail="Audio file not found")
    # Build download URL
    download_url = str(request.base_url) + f"audio-files/download/{audio_file.filename}"
    # Prepare metadata
    metadata = {
        "id": audio_file.id,
        "title": audio_file.title,
        "filename": audio_file.filename,
        "transcription": audio_file.transcription,
        "transcription_with_timestamps": audio_file.transcription_with_timestamps,
        "date": audio_file.date.isoformat() if audio_file.date else None,
        "author": audio_file.author,
        "category": audio_file.category,
        "user_id": audio_file.user_id,
        "created_at": audio_file.created_at.isoformat() if audio_file.created_at else None,
        "download_url": download_url
    }
    return JSONResponse(content=metadata)

@app.post("/notifications/mark-read/")
async def mark_notification_read(db: Session = Depends(get_db)):
    db.query(models.Notification).delete()
    db.commit()
    return {"detail": "All notifications cleared."}
