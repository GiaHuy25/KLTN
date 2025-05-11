from sqlalchemy.orm import Session
from models.models import Medicines as MedicineModel
from schemas.medicines import MedicineCreate, MedicineUpdate
from typing import List, Optional
from fastapi import UploadFile, HTTPException
from utils.file_handler import save_image, delete_image
from config.db import MEDIA_ROOT, DISEASE_IMAGE_DIRS
import logging

logger = logging.getLogger(__name__)

# --- CRUD Functions ---

def get_medicine_by_name(db: Session, medicine_name: str) -> Optional[int]:
    medicine = db.query(MedicineModel).filter(MedicineModel.name == medicine_name).first()
    return medicine.medicine_id if medicine else None


def get_medicines_by_disease_id(db: Session, disease_id: int) -> List[dict]:
    if disease_id is None:
        logger.warning("DiseaseID is None. Returning empty medicine list.")
        return []
    medicines = db.query(MedicineModel).filter(MedicineModel.disease_id == disease_id).all()
    return [
        {
            "id": medicine.medicine_id,
            "name": medicine.name,
            "price": float(medicine.price),
            "stock": medicine.stock,
            "image_url": medicine.image_url
        }
        for medicine in medicines
    ]


def get_medicine_details(db: Session, medicine_name: str) -> dict:
    medicine = db.query(MedicineModel).filter(MedicineModel.name == medicine_name).first()
    if not medicine:
        logger.warning(f"Medicine with name '{medicine_name}' not found.")
        return {
            "description": "Không có mô tả",
            "image_path": None
        }
    image_url = f"/media/{medicine.image_url}" if medicine.image_url else None
    return {
        "description": medicine.description or "Không có mô tả",
        "image_path": image_url
    }


def create_medicine(db: Session, medicine_data: MedicineCreate, image_file: Optional[UploadFile]) -> MedicineModel:
    saved_image_path = None
    if image_file:
        saved_image_path = save_image(image_file, medicine_data.name)
    db_medicine = MedicineModel(
        name=medicine_data.name,
        description=medicine_data.description,
        price=medicine_data.price,
        stock=medicine_data.stock_quantity,
        image_url=saved_image_path
    )
    db.add(db_medicine)
    db.commit()
    db.refresh(db_medicine)
    return db_medicine


def get_medicine(db: Session, medicine_id: int) -> MedicineModel:
    medicine = db.query(MedicineModel).filter(MedicineModel.medicine_id == medicine_id).first()
    if not medicine:
        logger.warning(f"Medicine with ID {medicine_id} not found.")
        raise HTTPException(status_code=404, detail="Medicine not found")
    return medicine


def update_medicine(db: Session, medicine_id: int, medicine_data: MedicineUpdate, image_file: Optional[UploadFile]) -> MedicineModel:
    db_medicine = get_medicine(db, medicine_id)
    update_data = medicine_data.dict(exclude_unset=True)
    new_image_path = None
    old_image_path = db_medicine.image_url
    if image_file:
        new_image_path = save_image(image_file, DISEASE_IMAGE_DIRS)
        update_data['image_url'] = new_image_path
    for key, value in update_data.items():
        setattr(db_medicine, key, value)
    db.add(db_medicine)
    db.commit()
    db.refresh(db_medicine)
    if new_image_path and old_image_path:
        delete_image(old_image_path)
    logger.info(f"Updated medicine {db_medicine.medicine_id}. New image path: {new_image_path}")
    return db_medicine


def delete_medicine(db: Session, medicine_id: int) -> int:
    db_medicine = get_medicine(db, medicine_id)
    image_path_to_delete = db_medicine.image_url
    db.delete(db_medicine)
    db.commit()
    delete_image(image_path_to_delete)
    logger.info(f"Deleted medicine {medicine_id}. Associated image path: {image_path_to_delete}")
    return medicine_id