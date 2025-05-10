from sqlalchemy.orm import Session
from typing import List, Optional
from fastapi import HTTPException, UploadFile
from schemas.medicines import MedicineCreate, MedicineUpdate
from models.models import Medicines
import os


def create_medicine(db: Session, medicine_data: MedicineCreate, file: UploadFile) -> Medicines:
    try:
        BASE_MEDIA_PATH = "media/medicines/"
        disease_map = {
            1: "PHYTOPHTHORA_PALMIVORA",
            2: "ALLOCARIDARA_ATTACK",
            3: "LEAF_BLIGHT",
            4: "LEAF_RHIZOCTONIA",
            5: "PHOMOPSIS_LEAF_SPOT",
            6: "HEALTHY_LEAF",
            7: "PHYTOPHTHORA_LEAF_BLIGHT",
            8: "LEAF_SPOT",
            9: "LEAF_ALGAL",
            10: "LEAF_COLLETOTRICHUM",
            11: "ALGAL_LEAF_SPOT"
        }
        folder_name = disease_map.get(medicine_data.disease_id, "Unknown")
        folder_path = os.path.join(BASE_MEDIA_PATH, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        db_medicine = Medicines(
            name=medicine_data.name,
            description=medicine_data.description,
            price=medicine_data.price,
            discounted_price=medicine_data.discounted_price,
            stock=medicine_data.stock,
            category_id=medicine_data.category_id,
            disease_id=medicine_data.disease_id,
            image_url="",
            is_freeship=medicine_data.is_freeship
        )
        db.add(db_medicine)
        db.commit()
        db.refresh(db_medicine)

        file_extension = os.path.splitext(file.filename)[1]
        file_name = f"{db_medicine.id}{file_extension}"
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        db_medicine.image_url = file_path
        db.commit()
        db.refresh(db_medicine)

        return db_medicine
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating medicine: {str(e)}")


def get_medicines(db: Session, skip: int = 0, limit: int = 100) -> List[Medicines]:
    return db.query(Medicines).offset(skip).limit(limit).all()


def get_medicine_by_id(db: Session, medicine_id: int) -> Optional[Medicines]:
    return db.query(Medicines).filter(Medicines.id == medicine_id).first()


def update_medicine(db: Session, medicine_id: int, medicine_data: MedicineUpdate) -> Optional[Medicines]:
    db_medicine = get_medicine_by_id(db, medicine_id)
    if db_medicine:
        db_medicine.name = medicine_data.name
        db_medicine.description = medicine_data.description
        db_medicine.price = medicine_data.price
        db_medicine.discounted_price = medicine_data.discounted_price
        db_medicine.stock = medicine_data.stock
        db_medicine.category_id = medicine_data.category_id
        db_medicine.disease_id = medicine_data.disease_id
        db_medicine.image_url = medicine_data.image_url
        db_medicine.is_freeship = medicine_data.is_freeship
        db.commit()
        db.refresh(db_medicine)
    return db_medicine


def delete_medicine(db: Session, medicine_id: int) -> Optional[Medicines]:
    db_medicine = get_medicine_by_id(db, medicine_id)
    if db_medicine:
        db.delete(db_medicine)
        db.commit()
        return db_medicine
    return None