from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict
from datetime import datetime

class UserBase(BaseModel):
    email: EmailStr
    username: str

class UserCreate(UserBase):
    password: str
    role: str = "user"  # Default role is "user" if not specified

class User(UserBase):
    id: int
    is_active: bool
    is_verified: bool
    roles: str
    created_at: datetime

    class Config:
        from_attributes = True

class AdminBase(BaseModel):
    email: EmailStr
    username: str

class AdminCreate(AdminBase):
    password: str

class Admin(AdminBase):
    id: int
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class PasswordReset(BaseModel):
    email: EmailStr

class PasswordUpdate(BaseModel):
    token: str
    new_password: str

class TimestampSegment(BaseModel):
    start_time: float  # Time in seconds
    end_time: float    # Time in seconds
    text: str         # The transcribed text for this segment

class TranscriptionWithTimestamps(BaseModel):
    segments: List[TimestampSegment]
    
    def to_string(self) -> str:
        """Convert the timestamped segments to a string format"""
        return "\n".join(
            f"[{int(seg.start_time//3600):02d}:{int((seg.start_time%3600)//60):02d}:{int(seg.start_time%60):02d}] {seg.text}"
            for seg in self.segments
        )
    
    @classmethod
    def from_string(cls, text: str) -> 'TranscriptionWithTimestamps':
        """Convert a string format back to timestamped segments"""
        segments = []
        for line in text.strip().split('\n'):
            if not line.strip():
                continue
            # Extract timestamp and text
            timestamp_str = line[1:line.find(']')]
            text = line[line.find(']')+1:].strip()
            
            # Convert timestamp to seconds
            h, m, s = map(int, timestamp_str.split(':'))
            start_time = h * 3600 + m * 60 + s
            
            segments.append(TimestampSegment(
                start_time=start_time,
                end_time=start_time + 5,  # Default 5 second duration if not specified
                text=text
            ))
        return cls(segments=segments)

class AudioFileBase(BaseModel):
    title: str
    transcription: str
    transcription_with_timestamps: Optional[str] = None  # Make it optional with default None
    filename: str
    date: datetime
    author: Optional[str] = None  # New field
    category: Optional[str] = None  # New field

class AudioFileCreate(AudioFileBase):
    pass

class AudioFile(AudioFileBase):
    id: int
    user_id: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True

class NotificationBase(BaseModel):
    type: str
    content: str
    extra_data: Optional[str] = None
    is_read: Optional[bool] = False

class NotificationCreate(NotificationBase):
    pass

class Notification(NotificationBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

class PublicUser(BaseModel):
    id: int
    email: EmailStr
    username: str
    is_active: bool
    is_verified: bool
    roles: str
    created_at: datetime

    class Config:
        from_attributes = True

class PublicAdmin(BaseModel):
    id: int
    email: EmailStr
    username: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True 