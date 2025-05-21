from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from config.db import get_db
from schemas.users import User, UserCreate, UserLogin, Token
from Service.users import create_user, get_user, get_users, authenticate_user
from jose import JWTError, jwt
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from models.models import User as UserModel  # Sử dụng User từ models.py

router = APIRouter(
    prefix="/users",
    tags=["Users"],
)

# Sử dụng HTTPBearer thay cho OAuth2PasswordBearer
security = HTTPBearer()

# Cấu hình JWT
SECRET_KEY = "your-secret-key"  # Thay bằng secret key thực tế
ALGORITHM = "HS256"

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)) -> UserModel:
    """
    Xác thực token và lấy thông tin user hiện tại.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(UserModel).filter(UserModel.email == email).first()
    if user is None:
        raise credentials_exception
    return user

async def get_current_admin(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    """
    Xác thực token và kiểm tra user có role là admin không.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(UserModel).filter(UserModel.email == email).first()
    if user is None:
        raise credentials_exception
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

@router.get("/", response_model=list[User], summary="Get all users")
async def read_users(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(get_db),
    current_admin: UserModel = Depends(get_current_admin)
):
    """
    Lấy danh sách tất cả user.
    Yêu cầu quyền admin.
    """
    try:
        users = get_users(db, skip=skip, limit=limit)
        return users
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving users: {str(e)}")

@router.get("/{id}", response_model=User, summary="Get user by ID")
async def read_user(
    id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    """
    Lấy thông tin user theo ID.
    - Người dùng thường chỉ có thể xem thông tin của chính họ.
    - Admin có thể xem thông tin của bất kỳ user nào.
    """
    if id != current_user.user_id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to access this user's information")

    try:
        user = get_user(db, user_id=id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving user: {str(e)}")

@router.post("/", response_model=User, summary="Create a new user")
async def create_user_endpoint(user: UserCreate, db: Session = Depends(get_db)):
    """
    Tạo một user mới.
    Không yêu cầu quyền admin.
    """
    try:
        existing_user = db.query(UserModel).filter(UserModel.email == user.email).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        result = create_user(db, user=user)
        return result["user"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating user: {str(e)}")

@router.post("/login", response_model=Token, summary="User login")
async def login(user: UserLogin, db: Session = Depends(get_db)):
    """
    Đăng nhập và nhận token.
    Không yêu cầu quyền admin.
    """
    try:
        result = authenticate_user(db, email=user.email, password=user.password)
        if result is None:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        return {
            "user_id": result["user_id"],
            "role": result["role"],
            "access_token": result["access_token"],
            "token_type": result["token_type"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")