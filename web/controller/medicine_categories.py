from sqlalchemy.orm import Session
from typing import List, Optional
from schemas.medicine_categories import MedicineCategoryCreate
from models.models import MedicineCategory


def create_medicine_category(db: Session, category_data: MedicineCategoryCreate) -> MedicineCategory:
    """
    Tạo một loại thuốc mới.
    """
    db_category = MedicineCategory(
        name=category_data.name,
        description=category_data.description
    )
    db.add(db_category)
    db.commit()
    db.refresh(db_category)
    return db_category


def get_medicine_categories(db: Session, skip: int = 0, limit: int = 100) -> List[MedicineCategory]:
    """
    Lấy danh sách các loại thuốc.
    """
    return db.query(MedicineCategory).offset(skip).limit(limit).all()


def get_medicine_category_by_id(db: Session, category_id: int) -> Optional[MedicineCategory]:
    """
    Lấy loại thuốc theo ID.
    """
    return db.query(MedicineCategory).filter(MedicineCategory.category_id == category_id).first()


def update_medicine_category(db: Session, category_id: int, category_data: MedicineCategoryCreate) -> Optional[MedicineCategory]:
    """
    Cập nhật thông tin loại thuốc.
    """
    db_category = get_medicine_category_by_id(db, category_id)
    if db_category:
        db_category.name = category_data.name
        db_category.description = category_data.description
        db.commit()
        db.refresh(db_category)
    return db_category


def delete_medicine_category(db: Session, category_id: int) -> Optional[MedicineCategory]:
    """
    Xóa loại thuốc theo ID.
    """
    db_category = get_medicine_category_by_id(db, category_id)
    if db_category:
        db.delete(db_category)
        db.commit()
        return db_category
    return None
