import os
import base64
import cv2
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from typing import List
from torchvision import transforms
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from sqlalchemy.exc import IntegrityError
from models.models import User as UserModel, Disease
from schemas.medicines import MedicineCreate, MedicineUpdate, Medicine, PaginatedMedicineResponse
from Service.medicines import create_medicine, get_medicine_by_id, update_medicine, delete_medicine, get_paginated_medicines
from config.db import get_db
from utils.paginator import paginate_dataframe
import logging

router = APIRouter(prefix="/medicines", tags=["Medicines"])
logger = logging.getLogger(__name__)
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH")

# Sử dụng HTTPBearer cho xác thực token
security = HTTPBearer()

# Cấu hình JWT
SECRET_KEY = "your-secret-key"  # Thay bằng secret key thực tế, phải khớp với user_router.py
ALGORITHM = "HS256"

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

@router.post("/", response_model=Medicine, status_code=status.HTTP_201_CREATED, summary="Tạo mới thuốc")
async def create_medicine_endpoint(
    medicine_data: MedicineCreate,
    image_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_admin: UserModel = Depends(get_current_admin)
):
    """
    Thêm thuốc mới vào cơ sở dữ liệu, bao gồm việc tải lên hình ảnh.
    Yêu cầu đăng nhập và role admin.
    """
    try:
        return create_medicine(db=db, medicine_data=medicine_data, file=image_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tạo thuốc: {str(e)}")

@router.get("/", response_model=PaginatedMedicineResponse, summary="Lấy danh sách thuốc với hình ảnh dạng Base64")
async def get_medicines_endpoint(page: int = 1, per_page: int = 10, db: Session = Depends(get_db)):
    """
    Lấy danh sách thuốc với hình ảnh dạng Base64.
    Không yêu cầu đăng nhập.
    """
    return get_paginated_medicines(db, page, per_page)

@router.get("/{medicine_id}", response_model=Medicine, summary="Lấy thông tin thuốc theo ID")
async def get_medicine_endpoint(medicine_id: int, db: Session = Depends(get_db)):
    """
    Lấy chi tiết thông tin của một thuốc dựa trên ID.
    Không yêu cầu đăng nhập.
    """
    try:
        medicine = get_medicine_by_id(db=db, medicine_id=medicine_id)
        if not medicine:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thuốc không tồn tại")
        return medicine
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy thông tin thuốc: {str(e)}")

@router.put("/{medicine_id}", response_model=Medicine, summary="Cập nhật thông tin thuốc")
async def update_medicine_endpoint(
    medicine_id: int,
    medicine_data: MedicineUpdate,
    db: Session = Depends(get_db),
    current_admin: UserModel = Depends(get_current_admin)
):
    try:
        logger.info(f"Updating medicine with ID {medicine_id} by admin {current_admin.email}")
        updated_medicine = update_medicine(db=db, medicine_id=medicine_id, medicine_data=medicine_data)
        if not updated_medicine:
            logger.warning(f"Medicine with ID {medicine_id} not found")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thuốc không tồn tại")
        logger.info(f"Medicine with ID {medicine_id} updated successfully")
        return updated_medicine
    except IntegrityError as e:
        logger.error(f"Database integrity error while updating medicine {medicine_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail="Dữ liệu không hợp lệ")
    except Exception as e:
        logger.error(f"Error updating medicine {medicine_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi khi cập nhật thuốc: {str(e)}")

@router.delete("/{medicine_id}", status_code=status.HTTP_200_OK, summary="Xóa thuốc theo ID")
async def delete_medicine_endpoint(
    medicine_id: int,
    db: Session = Depends(get_db),
    current_admin: UserModel = Depends(get_current_admin)
):
    """
    Xóa một thuốc khỏi cơ sở dữ liệu dựa trên ID.
    Yêu cầu đăng nhập và role admin.
    """
    try:
        deleted = delete_medicine(db=db, medicine_id=medicine_id)
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thuốc không tồn tại")
        return {"message": "Thuốc đã được xóa thành công", "deleted_id": medicine_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xóa thuốc: {str(e)}")