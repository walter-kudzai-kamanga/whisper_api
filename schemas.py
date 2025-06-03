from pydantic import BaseModel, EmailStr
from typing import Optional, List
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

class AudioFileBase(BaseModel):
    title: str
    filename: str
    category: Optional[str] = None
    author: Optional[str] = None
    duration: Optional[str] = None
    date: Optional[datetime] = None
    image: Optional[str] = None
    text_document: Optional[str] = None

class AudioFileCreate(AudioFileBase):
    pass

class AudioFile(AudioFileBase):
    id: int
    transcription: str
    user_id: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True

class NotificationBase(BaseModel):
    type: str
    content: str
    extra_data: Optional[str] = None

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