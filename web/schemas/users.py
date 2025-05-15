from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime

class UserBase(BaseModel):
    username: str = Field(..., min_length=1, max_length=50, description="Tên người dùng")
    email: EmailStr = Field(..., description="Địa chỉ email hợp lệ")
    full_name: Optional[str] = Field(None, max_length=100, description="Họ và tên")
    phone_number: Optional[str] = Field(None, max_length=15, description="Số điện thoại")
    address: Optional[str] = Field(None, description="Địa chỉ")

class UserCreate(UserBase):
    password: str = Field(..., min_length=8, description="Mật khẩu tối thiểu 8 ký tự")

class User(UserBase):
    user_id: int
    role: str
    created_at: datetime

    class Config:
        from_attributes = True

class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, description="Mật khẩu tối thiểu 8 ký tự")

class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: int

class TokenData(BaseModel):
    email: Optional[str] = None