from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from config.db import get_db
from schemas.users import User as UserSchema, UserCreate, UserLogin
from controller.users import create_user, get_user, get_users, authenticate_user

router = APIRouter(
    prefix="/users",
    tags=["Users"],
)

@router.get("/", response_model=list[UserSchema], summary="Get all users")
async def read_users(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    """
    Lấy danh sách tất cả người dùng.
    """
    try:
        users = get_users(db, skip=skip, limit=limit)
        return users
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving users: {str(e)}")

@router.get("/{id}", response_model=UserSchema, summary="Get user by ID")
async def read_user(id: int, db: Session = Depends(get_db)):
    """
    Lấy thông tin chi tiết của người dùng theo ID.
    """
    try:
        user = get_user(db, user_id=id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving user: {str(e)}")

@router.post("/", response_model=UserSchema, summary="Create a new user")
async def create_user_endpoint(user: UserCreate, db: Session = Depends(get_db)):
    """
    Tạo một người dùng mới.
    """
    try:
        # Kiểm tra xem email đã tồn tại chưa
        existing_user = db.query(UserSchema).filter(UserSchema.email == user.email).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        db_user = create_user(db, user=user)
        return db_user
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating user: {str(e)}")

@router.post("/login", summary="User login")
async def login(user: UserLogin, db: Session = Depends(get_db)):
    """
    Đăng nhập người dùng bằng email và mật khẩu.
    """
    try:
        db_user = authenticate_user(db, email=user.email, password=user.password)
        if db_user is None:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        return {
            "message": "Login successful",
            "user_id": db_user.user_id,
            "email": db_user.email,
            "full_name": db_user.full_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")
