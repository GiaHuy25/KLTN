from sqlalchemy.orm import Session
from models.models import User as UserModel
from schemas.users import UserCreate

def create_user(db: Session, user: UserCreate):
    db_user = UserModel(name=user.name, email=user.email, password=user.password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user(db: Session, user_id: int):
    return db.query(UserModel).filter(UserModel.id == user_id).first()

def get_users(db: Session, skip: int = 0, limit: int = 10):
    return db.query(UserModel).offset(skip).limit(limit).all()

def authenticate_user(db: Session, email: str, password: str):
    db_user = db.query(UserModel).filter(UserModel.email == email).first()
    if db_user is None or db_user.password != password:
        return None
    return db_user