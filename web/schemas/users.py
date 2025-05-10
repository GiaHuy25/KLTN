from pydantic import BaseModel, EmailStr, Field

# Cấu trúc cơ bản cho thông tin người dùng
class UserBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Tên người dùng")
    email: EmailStr = Field(..., description="Địa chỉ email hợp lệ")

# Schema để tạo người dùng mới
class UserCreate(UserBase):
    password: str = Field(..., min_length=8, description="Mật khẩu tối thiểu 8 ký tự")

# Schema để trả về thông tin người dùng
class User(UserBase):
    id: int

    class Config:
        from_attributes = True

# Schema cho quá trình đăng nhập
class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, description="Mật khẩu tối thiểu 8 ký tự")
