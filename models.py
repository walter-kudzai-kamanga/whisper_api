from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    roles = Column(String, default="user")  # Default role is "user"
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_activity = Column(DateTime(timezone=True), nullable=True)
    audio_files = relationship("AudioFile", back_populates="user")

class Admin(Base):
    __tablename__ = "admins"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_super_admin = Column(Boolean, default=False)  # Add super admin flag
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)

class AudioFile(Base):
    __tablename__ = "audio_files"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    transcription = Column(String, nullable=False)
    transcription_with_timestamps = Column(Text, nullable=True)  # Store transcription with timestamps
    date = Column(DateTime, nullable=False)
    author = Column(String, nullable=True)  # New field
    category = Column(String, nullable=True)  # New field
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="audio_files")

class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    type = Column(String)  # 'new_account', 'comment', 'feedback'
    content = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    extra_data = Column(Text, nullable=True)  # JSON string for additional data 
    is_read = Column(Boolean, default=False)  # Track if notification has been read 