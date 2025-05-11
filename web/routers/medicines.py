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
from models.models import Medicines
from schemas.medicines import MedicineCreate, MedicineUpdate, Medicine, PaginatedMedicineResponse
from Service.medicines import create_medicine, get_medicine_by_id, update_medicine, delete_medicine, get_paginated_medicines
from config.db import get_db
from utils.paginator import paginate_dataframe

router = APIRouter(prefix="/medicines", tags=["Medicines"])

load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH")

@router.post("/", response_model= Medicine, status_code=status.HTTP_201_CREATED, summary="Tạo mới thuốc")
def create_medicine_endpoint(
    medicine_data: MedicineCreate,
    image_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Thêm thuốc mới vào cơ sở dữ liệu, bao gồm việc tải lên hình ảnh.
    """
    try:
        return create_medicine(db=db, medicine_data=medicine_data, file=image_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tạo thuốc: {str(e)}")


@router.get("/", response_model=PaginatedMedicineResponse, summary="Lấy danh sách thuốc với hình ảnh dạng Base64")
def get_medicines_endpoint(page: int = 1, per_page: int = 10, db: Session = Depends(get_db)):
    return get_paginated_medicines(db, page, per_page)

@router.get("/{medicine_id}", response_model=Medicine, summary="Lấy thông tin thuốc theo ID")
def get_medicine_endpoint(medicine_id: int, db: Session = Depends(get_db)):
    """
    Lấy chi tiết thông tin của một thuốc dựa trên ID.
    """
    try:
        medicine = get_medicine_by_id(db=db, medicine_id=medicine_id)
        if not medicine:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thuốc không tồn tại")
        return medicine
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy thông tin thuốc: {str(e)}")


@router.put("/{medicine_id}", response_model=Medicine, summary="Cập nhật thông tin thuốc")
def update_medicine_endpoint(
    medicine_id: int,
    medicine_data: MedicineUpdate,
    db: Session = Depends(get_db)
):
    """
    Cập nhật thông tin của một thuốc dựa trên ID.
    """
    try:
        updated_medicine = update_medicine(db=db, medicine_id=medicine_id, medicine_data=medicine_data)
        if not updated_medicine:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thuốc không tồn tại")
        return updated_medicine
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi cập nhật thuốc: {str(e)}")


@router.delete("/{medicine_id}", status_code=status.HTTP_200_OK, summary="Xóa thuốc theo ID")
def delete_medicine_endpoint(medicine_id: int, db: Session = Depends(get_db)):
    """
    Xóa một thuốc khỏi cơ sở dữ liệu dựa trên ID.
    """
    try:
        deleted = delete_medicine(db=db, medicine_id=medicine_id)
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thuốc không tồn tại")
        return {"message": "Thuốc đã được xóa thành công", "deleted_id": medicine_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xóa thuốc: {str(e)}")
