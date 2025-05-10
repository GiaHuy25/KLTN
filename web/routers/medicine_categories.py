from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session
import pandas as pd

from config.db import get_db
from schemas.medicine_categories import MedicineCategory as CategorySchema, MedicineCategoryCreate
from controller.medicine_categories import (
    create_medicine_category,
    get_medicine_category_by_id,
    get_medicine_categories,
    update_medicine_category,
    delete_medicine_category
)
from utils.paginator import paginate_dataframe

router = APIRouter(prefix="/categories", tags=["Medicine Categories"])

@router.get("/", status_code=status.HTTP_200_OK, summary="Lấy danh sách phân loại thuốc")
async def read_categories(request: Request, db: Session = Depends(get_db)):
    """
    Lấy danh sách phân loại thuốc, có phân trang.
    """
    try:
        page = int(request.query_params.get("page", 1))
        per_page = int(request.query_params.get("per_page", 10))

        all_categories = get_medicine_categories(db, skip=0, limit=1000)
        df = pd.DataFrame([{
            "id": cat.category_id,
            "name": cat.name,
            "description": cat.description,
            "created_at": cat.created_at
        } for cat in all_categories])

        return paginate_dataframe(df, page, per_page)

    except ValueError:
        raise HTTPException(status_code=422, detail="Page and per_page must be integers")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{id}", response_model=CategorySchema, summary="Lấy chi tiết phân loại thuốc")
async def read_category(id: int, db: Session = Depends(get_db)):
    """
    Lấy thông tin chi tiết một phân loại thuốc theo ID.
    """
    category = get_medicine_category_by_id(db, category_id=id)
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    return category

@router.post("/", response_model=CategorySchema, status_code=status.HTTP_201_CREATED, summary="Tạo mới phân loại thuốc")
async def create_category_endpoint(category: MedicineCategoryCreate, db: Session = Depends(get_db)):
    """
    Tạo mới một phân loại thuốc.
    """
    return create_medicine_category(db, category_data=category)

@router.put("/{id}", response_model=CategorySchema, summary="Cập nhật phân loại thuốc")
async def update_category_endpoint(id: int, category: MedicineCategoryCreate, db: Session = Depends(get_db)):
    """
    Cập nhật thông tin một phân loại thuốc.
    """
    updated = update_medicine_category(db, category_id=id, category_data=category)
    if not updated:
        raise HTTPException(status_code=404, detail="Category not found")
    return updated

@router.delete("/{id}", response_model=CategorySchema, summary="Xóa phân loại thuốc")
async def delete_category_endpoint(id: int, db: Session = Depends(get_db)):
    """
    Xóa một phân loại thuốc.
    """
    deleted = delete_medicine_category(db, category_id=id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Category not found")
    return deleted
